import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import pyrealsense2 as rs
import rospy

np.random.seed(16)
tf.random.set_seed(16)

class RetinaNet(keras.Model):
    """A subclassed Keras model implementing the RetinaNet architecture.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
        Currently supports ResNet50 only.
    """

    def __init__(self, num_classes, backbone=None, **kwargs):
        super(RetinaNet, self).__init__(name="RetinaNet", **kwargs)
        self.fpn = FeaturePyramid(backbone)
        self.num_classes = num_classes

        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        self.cls_head = build_head(9 * num_classes, prior_probability)
        self.box_head = build_head(9 * 4, "zeros")

    def call(self, image, training=False):
        features = self.fpn(image, training=training)
        N = tf.shape(image)[0]
        cls_outputs = []
        box_outputs = []
        for feature in features:
            box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
            cls_outputs.append(
                tf.reshape(self.cls_head(feature), [N, -1, self.num_classes])
            )
        cls_outputs = tf.concat(cls_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)
        return tf.concat([box_outputs, cls_outputs], axis=-1)

class FeaturePyramid(keras.layers.Layer):
    """Builds the Feature Pyramid with the feature maps from the backbone.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
        Currently supports ResNet50 only.
    """

    def __init__(self, backbone=None, **kwargs):
        super(FeaturePyramid, self).__init__(name="FeaturePyramid", **kwargs)
        self.backbone = backbone# if backbone else get_backbone()
        self.conv_c3_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c4_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c5_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c3_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c4_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c5_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c6_3x3 = keras.layers.Conv2D(256, 3, 2, "same")
        self.conv_c7_3x3 = keras.layers.Conv2D(256, 3, 2, "same")
        self.upsample_2x = keras.layers.UpSampling2D(2)

    def call(self, images, training=False):
        c3_output, c4_output, c5_output = self.backbone(images, training=training)
        p3_output = self.conv_c3_1x1(c3_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p5_output = self.conv_c5_1x1(c5_output)
        p4_output = p4_output + self.upsample_2x(p5_output)
        p3_output = p3_output + self.upsample_2x(p4_output)
        p3_output = self.conv_c3_3x3(p3_output)
        p4_output = self.conv_c4_3x3(p4_output)
        p5_output = self.conv_c5_3x3(p5_output)
        p6_output = self.conv_c6_3x3(c5_output)
        p7_output = self.conv_c7_3x3(tf.nn.relu(p6_output))
        return p3_output, p4_output, p5_output, p6_output, p7_output

def build_head(output_filters, bias_init):
    """Builds the class/box predictions head.

    Arguments:
      output_filters: Number of convolution filters in the final layer.
      bias_init: Bias Initializer for the final convolution layer.

    Returns:
      A keras sequential model representing either the classification
        or the box regression head depending on `output_filters`.
    """
    head = keras.Sequential([keras.Input(shape=[None, None, 256])])
    kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
    for _ in range(4):
        head.add(
            keras.layers.Conv2D(256, 3, padding="same", kernel_initializer=kernel_init)
        )
        head.add(keras.layers.ReLU())
    head.add(
        keras.layers.Conv2D(
            output_filters,
            3,
            1,
            padding="same",
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
        )
    )
    return head

class DecodePredictions(tf.keras.layers.Layer):
    """A Keras layer that decodes predictions of the RetinaNet model.

    Attributes:
      num_classes: Number of classes in the dataset
      confidence_threshold: Minimum class probability, below which detections
        are pruned.
      nms_iou_threshold: IOU threshold for the NMS operation
      max_detections_per_class: Maximum number of detections to retain per
       class.
      max_detections: Maximum number of detections to retain across all
        classes.
      box_variance: The scaling factors used to scale the bounding box
        predictions.
    """

    def __init__(
        self,
        num_classes=1,
        confidence_threshold=0.05,
        nms_iou_threshold=0.5,
        max_detections_per_class=6,
        max_detections=6,
        box_variance=[0.1, 0.1, 0.2, 0.2],
        **kwargs
    ):
        super(DecodePredictions, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections

        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions * self._box_variance
        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
        boxes_transformed = convert_to_corners(boxes)
        return boxes_transformed

    def call(self, images, predictions):
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        box_predictions = predictions[:, :, :4]
        cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:])
        boxes = self._decode_box_predictions(anchor_boxes[None, ...], box_predictions)

        return tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            cls_predictions,
            self.max_detections_per_class,
            self.max_detections,
            self.nms_iou_threshold,
            self.confidence_threshold,
            clip_boxes=False,
        )

class AnchorBox:
    """Generates anchor boxes.

    This class has operations to generate anchor boxes for feature maps at
    strides `[8, 16, 32, 64, 128]`. Where each anchor each box is of the
    format `[x, y, width, height]`.

    Attributes:
      aspect_ratios: A list of float values representing the aspect ratios of
        the anchor boxes at each location on the feature map
      scales: A list of float values representing the scale of the anchor boxes
        at each location on the feature map.
      num_anchors: The number of anchor boxes at each location on feature map
      areas: A list of float values representing the areas of the anchor
        boxes for each feature map in the feature pyramid.
      strides: A list of float value representing the strides for each feature
        map in the feature pyramid.
    """

    def __init__(self):
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]

        self._num_anchors = len(self.aspect_ratios) * len(self.scales)
        self._strides = [2 ** i for i in range(3, 8)]
        self._areas = [x ** 2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        """Computes anchor box dimensions for all ratios and scales at all levels
        of the feature pyramid.
        """
        anchor_dims_all = []
        for area in self._areas:
            anchor_dims = []
            for ratio in self.aspect_ratios:
                anchor_height = tf.math.sqrt(area / ratio)
                anchor_width = area / anchor_height
                dims = tf.reshape(
                    tf.stack([anchor_width, anchor_height], axis=-1), [1, 1, 2]
                )
                for scale in self.scales:
                    anchor_dims.append(scale * dims)
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
        return anchor_dims_all

    def _get_anchors(self, feature_height, feature_width, level):
        """Generates anchor boxes for a given feature map size and level

        Arguments:
          feature_height: An integer representing the height of the feature map.
          feature_width: An integer representing the width of the feature map.
          level: An integer representing the level of the feature map in the
            feature pyramid.

        Returns:
          anchor boxes with the shape
          `(feature_height * feature_width * num_anchors, 4)`
        """
        rx = tf.range(feature_width, dtype=tf.float32) + 0.5
        ry = tf.range(feature_height, dtype=tf.float32) + 0.5
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * self._strides[level - 3]
        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])
        dims = tf.tile(
            self._anchor_dims[level - 3], [feature_height, feature_width, 1, 1]
        )
        anchors = tf.concat([centers, dims], axis=-1)
        return tf.reshape(
            anchors, [feature_height * feature_width * self._num_anchors, 4]
        )

    def get_anchors(self, image_height, image_width):
        """Generates anchor boxes for all the feature maps of the feature pyramid.

        Arguments:
          image_height: Height of the input image.
          image_width: Width of the input image.

        Returns:
          anchor boxes for all the feature maps, stacked as a single tensor
            with shape `(total_anchors, 4)`
        """
        anchors = [
            self._get_anchors(
                tf.math.ceil(image_height / 2 ** i),
                tf.math.ceil(image_width / 2 ** i),
                i,
            )
            for i in range(3, 8)
        ]
        return tf.concat(anchors, axis=0)

def convert_to_corners(boxes):
    """Changes the box format to corner coordinates

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[x, y, width, height]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )

def create_bboxed_images(image, bboxes_pred, desired_size=510):
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    old_size = image.size

    ratio = float(desired_size)/max(old_size)
    black_bar_size = (desired_size - (min(old_size)*ratio))/2
    
    cropped_images = []
    bboxes = []
    
    if old_size[0] < old_size[1]:
        for i in range(len(bboxes_pred)):
            x1 = int((bboxes_pred[i][0] - black_bar_size)/ratio)
            x2 = int((bboxes_pred[i][2] - black_bar_size)/ratio)
            y1 = int((bboxes_pred[i][1])/ratio)
            y2 = int((bboxes_pred[i][3])/ratio)
            
            bbox = [x1,y1,x2,y2]
            bboxes.append(bbox)
            cropped_im = image.crop(bbox)
            cropped_images.append(cropped_im)
    else:
        for i in range(len(bboxes_pred)):
            y1 = int((bboxes_pred[i][1] - black_bar_size)/ratio)
            y2 = int((bboxes_pred[i][3] - black_bar_size)/ratio)
            x1 = int((bboxes_pred[i][0])/ratio)
            x2 = int((bboxes_pred[i][2])/ratio)

            bbox = [x1,y1,x2,y2]
            bboxes.append(bbox)
            cropped_im = image.crop(bbox)
            cropped_images.append(cropped_im)

    return cropped_images, bboxes

def prepare_image(image):
    image, _, ratio = resize_and_pad_image(image, jitter=None)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio

def resize_and_pad_image(
    image, min_side=800.0, max_side=1333.0, jitter=[640, 1024], stride=128.0
):
    """Resizes and pads image while preserving aspect ratio.

    1. Resizes images so that the shorter side is equal to `min_side`
    2. If the longer side is greater than `max_side`, then resize the image
      with longer side equal to `max_side`
    3. Pad with zeros on right and bottom to make the image shape divisible by
    `stride`

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      min_side: The shorter side of the image is resized to this value, if
        `jitter` is set to None.
      max_side: If the longer side of the image exceeds this value after
        resizing, the image is resized such that the longer side now equals to
        this value.
      jitter: A list of floats containing minimum and maximum size for scale
        jittering. If available, the shorter side of the image will be
        resized to a random value in this range.
      stride: The stride of the smallest feature map in the feature pyramid.
        Can be calculated using `image_size / feature_map_size`.

    Returns:
      image: Resized and padded image.
      image_shape: Shape of the image before padding.
      ratio: The scaling factor used to resize the image
    """
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    if jitter is not None:
        min_side = tf.random.uniform((), jitter[0], jitter[1], dtype=tf.float32)
    ratio = min_side / tf.reduce_min(image_shape)
    if ratio * tf.reduce_max(image_shape) > max_side:
        ratio = max_side / tf.reduce_max(image_shape)
    image_shape = ratio * image_shape
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
    padded_image_shape = tf.cast(
        tf.math.ceil(image_shape / stride) * stride, dtype=tf.int32
    )
    image = tf.image.pad_to_bounding_box(
        image, 0, 0, padded_image_shape[0], padded_image_shape[1]
    )
    return image, image_shape, ratio

def resize_image(image, desired_size=510):

    image = Image.fromarray(image.astype('uint8'), 'RGB')
    old_size = image.size  # old_size[0] is in (width, height) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    image = image.resize(new_size, Image.ANTIALIAS)
    # create a new image and paste the resized on it

    new_image = Image.new("RGB", (desired_size, desired_size))
    new_image.paste(image, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))
    
    return new_image
  
def get_detection_model(pwd_model=None):
  num_classes = 1
  resnet50_backbone = get_backbone()

  model = RetinaNet(num_classes, resnet50_backbone)
  try:
    latest_checkpoint = tf.train.latest_checkpoint(pwd_model)
    model.load_weights(latest_checkpoint)
    print("Loaded weights")
  except:
    print("Could not find weights, using random weights")
  image = tf.keras.Input(shape=[None, None, 3], name="image")
  predictions = model(image, training=False)
  detections = DecodePredictions(confidence_threshold=0.8)(image, predictions)
  inference_model = tf.keras.Model(inputs=image, outputs=detections)
  return inference_model

def get_backbone():
  """Builds ResNet50 with pre-trained imagenet weights"""
  backbone = keras.applications.ResNet50(include_top=False, input_shape=[None, None, 3])
  c3_output, c4_output, c5_output = [backbone.get_layer(layer_name).output for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]]
  return keras.Model(inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output])

def predict_truss(image, inference_model):
  desired_size = 510
  # Resize image
  image_resized = resize_image(image, desired_size=desired_size)

  # Prediction
  img = tf.cast(np.array(image_resized), dtype=tf.float32)
  input_image, ratio = prepare_image(img)
  detections = inference_model.predict(input_image)
  num_detections = detections.valid_detections[0]
  print(f'Prediction: {num_detections} detections')

  class_names = [int(x) for x in detections.nmsed_classes[0][:num_detections]]
  bboxes_pred = detections.nmsed_boxes[0][:num_detections] / ratio

  return num_detections, bboxes_pred

def camera_info2rs_intrinsics(camera_info):

    # init object
    rs_intrinsics = rs.intrinsics()

    # dimensions
    rs_intrinsics.width = camera_info.width
    rs_intrinsics.height = camera_info.height

    # principal point coordinates
    rs_intrinsics.ppx = camera_info.K[2]
    rs_intrinsics.ppy = camera_info.K[5]

    # focal point
    rs_intrinsics.fx = camera_info.K[0]
    rs_intrinsics.fy = camera_info.K[4]

    return rs_intrinsics
  
class DepthImageFilter(object):
    """DepthImageFilter"""

    def __init__(self, image, intrinsics, patch_size=5, node_name="depth interface"):
        self.image = image
        self.intrinsics = intrinsics
        self.node_name = node_name
        self.patch_radius = int((patch_size - 1) / 2)

        # patch
        self.min_row = 0
        self.max_row = self.image.shape[0]
        self.min_col = 0
        self.max_col = self.image.shape[1]

    def generate_patch(self, row, col):
        """Returns a patch which can be used for filtering an image"""
        row = int(round(row))
        col = int(round(col))

        row_start = max([row - self.patch_radius, self.min_row])
        row_end = min([row + self.patch_radius, self.max_row - 1]) + 1

        col_start = max([col - self.patch_radius, self.min_col])
        col_end = min([col + self.patch_radius, self.max_col - 1]) + 1

        rows = np.arange(row_start, row_end)
        cols = np.arange(col_start, col_end)
        return rows, cols

    def deproject(self, row, col, depth=None, segment=None):
        """
        Deproject a 2D coordinate to a 3D point using the depth image, if an depth is provided the depth image is
        ignored.
        """

        if depth is None:
            depth = self.get_depth(row, col, segment=segment)

        if np.isnan(depth):
            rospy.logwarn("[OBJECT DETECTION] Computed depth is nan, can not compute point!")
            return 3 * [np.nan]

        point = rs.rs2_deproject_pixel_to_point(self.intrinsics, [col, row], depth)
        rospy.logdebug("[{0}] Point obtained from depth image {1}".format(self.node_name, np.array(point)))
        return point
      
    def get_depth(self, row, col, segment=None):
      """get the filtered depth from an depth image at location row, col"""
      rows, cols = self.generate_patch(row, col)

      if segment is None:
          depth_patch = self.image[rows[:, np.newaxis], cols]
      else:
          depth_patch = self.image[segment > 0]

      non_zero = np.nonzero(depth_patch)
      depth_patch_non_zero = depth_patch[non_zero]
      #return np.median(depth_patch_non_zero)
      return np.percentile(depth_patch_non_zero, 10)