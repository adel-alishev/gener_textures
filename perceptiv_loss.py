import tensorflow as tf
import IPython.display as display
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def clip_0_1(image):
    """
    Мы хотим уметь отображать нашу полученную картинку, а для этого ее значения должны
    находится в промежутке от 0 до 1. Наш алгоритм оптимизации этого нигде не учитывает
    поэтому к полученному изображению мы будем применять "обрезку" по значению

    """
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
def load_img(path_to_img, max_dim=512):
    """
    Данная функция считывает изображение с диска и приводит его к такому размеру,
    чтобы бОльшая сторона была равна max_dim пикселей.

    Для считывания изображения воспользуемся функциями tensorflow.
    """
    img = tf.io.read_file(path_to_img)  # считываени файла
    img = tf.image.decode_image(img, channels=3)  # декодинг
    img = tf.image.convert_image_dtype(img, tf.float32)  # uint8 -> float32, 255 -> 1

    shape = img.numpy().shape[:-1]
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tuple((np.array(shape) * scale).astype(np.int32))

    img = tf.image.resize(img, new_shape)  # изменение размера
    img = img[tf.newaxis, :]  # добавляем batch dimension
    return img
def imshow(image, title=None):
    """
    Функция для отрисовки изображения
    """
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)
    plt.show()

url = 'https://happywall-img-gallery.imgix.net/2657/grey_pebble_simplicity_display.jpg'
image_path = tf.keras.utils.get_file('stones.jpg', url)

content_image = load_img(image_path)
imshow(content_image, 'Image')
content_image.numpy().max(), content_image.numpy().shape # убедимся, что картинка нужного размера, а также значения
                                                         # лежат в промежутке от 0 до 1

def get_vgg_layers_model(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    for layer in vgg.layers:
        print(layer.name)
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model
# get_vgg_layers_model(["block3_conv1"]).summary()
class FeatureExtractor:
    def __init__(self, layers):
        self.vgg_outputs_model = get_vgg_layers_model(layers)
        self.vgg_outputs_model.trainable = False
        self.content_layers = layers

    def __call__(self, inputs):
        """
        На входе 4х мерный тензор (картинка). Значения пикселей ограничены 0..1!

        На выходе: {"имя слоя": тензор выхода этого слоя}
        """
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs * 255.)  # VGG препроцессинг
        outputs = self.vgg_outputs_model(preprocessed_input)
        features_dict = {name: value for name, value in zip(self.content_layers, outputs)}

        return features_dict
def loss(image):
    """
    Получаем картинку, вычисляем признаки с помощью класса FeatureExtractor.
    Сравниваем их с target_features с помощью MeanSquaredError.
    """
    current_features = extractor(image)
    loss = tf.add_n([tf.keras.losses.MeanSquaredError()(current_features[name], target_features[name])
                                     for name in target_features.keys()])
    loss *= 1. / len(target_features.keys())

    # для того чтобы результаты были больше похожи на настоящую картинку -- добавим регуляризацию
    # в реальных картинках цвета меняются плавно и нет цветового шума (шумные цветные пиксели поверх картинки)
    # при оптимизации мы часто будем получать такие результаты -- чтобы их уменьшать будуем штрафовать за такие резкие перепады цветов.
    # tota_variation -- нам в этом поможет
    loss += tf.image.total_variation(image)*1e-2
    return loss
def train_step(image, loss_func, optimizer):
    """
    Шаг оптимизации мы реализуем вручную (без .fit()). Такая реализация будет
    нам полезна в дальнейшем.

    """

    with tf.GradientTape() as tape:  # "записываем" градиенты для дальнейшего использования
        loss = loss_func(image)
    grad = tape.gradient(loss, image)  # dLoss/dImage
    optimizer.apply_gradients(
        [(grad, image)])  # шаг градиентного спуск. в случае  GD: image = image - lambda*dLoss/dImage
    # картинка после этого шага изменилась!
    image.assign(clip_0_1(image))  # ~ image = clip_0_1(image), "обрезаем" неправильные значения
    return loss.numpy()
def show_pair(original, generated, title=None):
    imshow(original, 'Original Image')
    imshow(generated, title)

# block2_conv2, block4_conv1, block5_conv3
#url = 'https://happywall-img-gallery.imgix.net/2657/grey_pebble_simplicity_display.jpg'
url = "https://cdn.britannica.com/s:500x350/86/170586-120-7E23E561.jpg"
image_path = tf.keras.utils.get_file('taj.jpg', url)
content_image = load_img(image_path, 256)

feature_layers = ['block5_conv3']
extractor = FeatureExtractor(feature_layers)
target_features = extractor(content_image)

image = tf.Variable(np.random.rand(*content_image.numpy().shape).astype(np.float32))
opt = tf.keras.optimizers.Adam(learning_rate=0.2, beta_1=0.99, epsilon=1e-1)


# сделаем шаг оптимизации -- убедимся что все работает без ошибок.
train_step(image, loss_func=loss, optimizer=opt)
show_pair(content_image, image)

start = time.time()

epochs = 10
steps_per_epoch = 100

step = 0
for n in range(epochs):
  for m in tqdm(range(steps_per_epoch)):
    step += 1
    train_step(image, loss_func=loss, optimizer=opt)

  display.clear_output(wait=True)
  show_pair(content_image, image, f"Generated Image. Optimized for {feature_layers[0]}. Train step: {step}")
  plt.show()

end = time.time()
print("Total time: {:.1f}".format(end-start))
display.clear_output(wait=True)
show_pair(content_image, image, f"Generated Image. Optimize for {feature_layers[0]}.")
plt.savefig(f"result_{feature_layers[0]}.png")