import tensorflow as tf
import matplotlib.image as mpimg
import skvideo
import numpy as np
from PIL import Image



class VideoExtractor(object):
    """Select uniformly distributed frames and extract its VGG feature."""

    def __init__(self, frame_num, sess):
        """Load resnet model.

        Args:
            frame_num: number of frames per video.
            sess: tf.Session()
        """
        tf.keras.backend.set_session(sess)
        self.frame_num = frame_num
        # self.inputs = tf.placeholder(tf.float32, [self.frame_num, 224, 224, 3])
        # self.resnet_v2 = tf.keras.applications.InceptionResNetV2(include_top=False,
        #                                                          input_shape=(224, 224, 3),
        #                                                          pooling='max')
        self.resnet_v2 = tf.keras.applications.ResNet50(False, input_shape=(224, 224, 3), pooling='max')

    def _select_frames(self, path):
        """Select representative frames for video.

        Ignore some frames both at begin and end of video.

        Args:
            path: Path of video.
        Returns:
            frames: list of frames.
        """
        frames = list()
        # video_info = skvideo.io.ffprobe(path)
        video_data = skvideo.io.vread(path)
        total_frames = video_data.shape[0]
        # Ignore some frame at begin and end.
        for i in np.linspace(0, total_frames, self.frame_num + 2)[1:self.frame_num + 1]:
            frame_data = video_data[int(i)]
            img = Image.fromarray(frame_data)
            img = img.resize((224, 224), Image.BILINEAR)
            frame_data = np.array(img)
            frames.append(frame_data)
        return frames

    def extract(self, path):
        """Get VGG fc7 activations as representation for video.

        Args:
            path: Path of video.
        Returns:
            feature: [batch_size, 4096]
        """
        frames = self._select_frames(path)
        feature = self.resnet_v2.predict(np.array(frames), batch_size=len(frames))
        return feature


if __name__ == '__main__':
    def m():
        img = Image.open(open('./test.jpg', 'rb'))
        img = img.resize((224, 224), Image.BILINEAR)
        x = np.array(img)
        model_a = tf.keras.applications.ResNet50(False, input_shape=(224, 224, 3), pooling='max')
        x = x.reshape((1, 224, 224, 3))
        info = model_a.predict(x)
        print(info.shape)
        # tf.keras.applications.inception_resnet_v2.preprocess_input(x)

    # for test
    m()
