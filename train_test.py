import tensorflow as tf
import os
import pathlib
import time
import datetime
import imageio

from tensorflow import keras
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from IPython import display
from model import Generator, Discriminator, generator_loss, discriminator_loss
from datagenerator import *
from config import PLOT_PATH, IMG_DIR, ENCODED_PATH, IMAGES
from preprocessing import resize, random_crop, random_jitter, normalize

"""START OF LOADING AND PREPROCESSING IMAGE PART"""

"""
THIS IS TO LOAD ALL AND JUST THE rgb_tot_white IMAGES. I DON'T NEED IT (AT LEAST FOR THE MOMENT)

print("Loading data_w..")
dataset = DatasetRGB(rgb_dir=IMG_DIR)
dataset.prepare_data()  # splitta dataset in train, test, val
print("Done!")
(X_train), (X_val), (_) = dataset.data
RGB_train = X_train["RGB"]  # This is the batch of images for the training
RGB_val = X_val["RGB"]      # This is the batch of images for the validation
print('RGB train :   ', np.shape(RGB_train))    # output: (752, 256, 256, 3)
print('RGB val :   ', np.shape(RGB_val))        # output: (5, 256, 256, 3)
# I plot one image just to show
plt.imshow(RGB_train[1])
plt.show()
# From this moment, I have training and validation dataset respectively in RGB_train, RGB_val
"""

PATH = IMG_DIR

# sample_image = tf.io.read_file(PATH + "2_weight_and_strawberry.png")
# sample_image = tf.io.decode_png(sample_image)
# print(sample_image.shape)


def load(image_file):
    # Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)

    # Split each image tensor into two tensors:
    # - one with a real building facade image
    # - one with an architecture label image
    w = tf.shape(image)[1]
    w = w // 2
    input_image = image[:, w:, :]
    real_image = image[:, :w, :]

    # Convert both images to float32 tensors
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


# inp, re = load(PATH + "2_weight_and_strawberry.png")

""" We here start to apply the preprocessing to the images """
BUFFER_SIZE = 175  # the weight+strawberry dataset consists of 175 images
BATCH_SIZE = 1  # in the paper, a batch_size of 1 produced better results
# each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256


# def resize
# def random_crop
# def normalize
# @tf.function()  --> WATCH OUT, i think this one needs to be un-commented
# def random_jitter


# I now omit the part where it inspects some preprocessed output

# Let's now define two helper functions that load and preprocess the training and test sets
# I have already loaded the images, so I should just perform random_jitter and normalize functions,
# separately for train and test set (there is the difference of input e real image)


def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                     IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


"""Build an input pipeline with tf.data_w ( link to guide: https://www.tensorflow.org/guide/data )"""
# A dataset of all files matching one or more glob patterns.
# The default behavior of this method is to return filenames in a non-deterministic random shuffled order
train_dataset = tf.data.Dataset.list_files(PATH + "train/*.png")
# produces a new dataset by applying a given function (f) to each element of the input dataset
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
# maintains a fixed-size buffer and chooses the next element uniformly at random from that buffer
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
# batches BATCH_SIZE elements of a dataset into a single element
train_dataset = train_dataset.batch(BATCH_SIZE)

try:
    test_dataset = tf.data.Dataset.list_files(PATH + "test/*.png")
except tf.errors.InvalidArgumentError:
    test_dataset = tf.data.Dataset.list_files(PATH + "val/*.png")
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

"""END OF LOADING AND PREPROCESSING IMAGE PART"""

OUTPUT_CHANNELS = 3
generator = Generator()
discriminator = Discriminator()

LAMBDA = 100

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

"""Define the optimizers and a checkpoint-saver"""
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

"""Generate Images"""


def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


"""Training"""
log_dir = "logs/"

summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))  # YearMonthDay-HoursMinutesSeconds


@tf.function
def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step // 1000)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step // 1000)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step // 1000)
        tf.summary.scalar('disc_loss', disc_loss, step=step // 1000)


def fit(train_ds, test_ds, steps):
    example_input, example_target = next(iter(test_ds.take(1)))
    start = time.time()

    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        if (step) % 1000 == 0:
            display.clear_output(wait=True)

            if step != 0:
                print(f'Time taken for 1000 steps: {time.time() - start:.2f} sec\n')

            start = time.time()

            generate_images(generator, example_input, example_target)
            print(f"Step: {step // 1000}k")

        train_step(input_image, target, step)

        # Training step
        if (step + 1) % 10 == 0:
            print('.', end='', flush=True)

        # Save (checkpoint) the model every 5k steps
        if (step + 1) % 5000 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


# %load_ext tensorboard
# %tensorboard --logdir {log_dir}

fit(train_dataset, test_dataset, steps=10)

"""Restore the latest checkpoint and test the network"""
# !ls {checkpoint_dir}

# Restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

"""Generate some images using the test set"""
# Run the trained model on a few examples from the test set
for inp, tar in test_dataset.take(5):
    generate_images(generator, inp, tar)
