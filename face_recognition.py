import argparse
import os
import common
import json
import tensorflow as tf
from custom_iresnet_v2 import *
import tensorflow as tf
import common
import logging
import numpy as np
import cv2
import mtcnn
from sklearn.calibration import LabelEncoder
from keras import losses
from keras.models import Model
from keras.layers import Dense, Input
from keras.models import load_model
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras import backend as K
from keras.optimizers.legacy import SGD, Adam, Nadam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

networks = ['CustomInceptionResNetV2','InceptionResNetV2']

class FaceRecognition(common.COMMON):
    model = None
    ckpt_exist = False
    def __init__(self, args):
        self.dataset = args.dataset
        self.checkpoint = args.checkpoint
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.learning_rate = args.lr
        self.optimizer = args.opt
        self.network = args.network
        self.image_size = args.image_size
        self.channel = args.channel

        if not os.path.exists(self.checkpoint):
            os.makedirs(self.checkpoint)
        else:
            files = os.listdir(self.checkpoint)
            self.ckpt_exist = 'model.h5' in files
        self.class_num = self.get_classes_count()

    def get_classes_count(self):
        return len(os.listdir(os.path.join(self.dataset, 'train')))
    
    def normalize(img):
        mean, std = img.mean(), img.std()
        return (img - mean) / std

    def get_datasets(self):
        face_detector = mtcnn.MTCNN()
        required_shape = (299,299)
        faces = []
        labels = []
        for face_names in os.listdir(self.dataset):
            person_dir = os.path.join(self.dataset, face_names)

            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir,image_name)

                img_BGR = cv2.imread(image_path)
                img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

                x = face_detector.detect_faces(img_RGB)
                x1, y1, width, height = x[0]['box']
                x1, y1 = abs(x1) , abs(y1)
                x2, y2 = x1+width , y1+height
                face = img_RGB[y1:y2 , x1:x2]
                
                face = self.normalize(face)
                face = cv2.resize(face, required_shape)
                # face_d = np.expand_dims(face, axis=0)

                faces.append(face)
                labels.append(face_names)

        return np.array(faces), np.array(labels)

    def build_model(self):
        input_size = (self.image_size, self.image_size, self.channel)
        input = Input(shape=(299, 299, 3))
        self.log_info('Model: {}'.format(self.network))
        if self.network == 'CustomInceptionResNetV2':
            self.model = CustomInceptionResNetV2(input_size, self.class_num)()
        else:
            base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=input, input_shape=input_size, pooling='avg')

        if self.network not in ['CustomInceptionResNetV2']:
            identifiers = Dense(units=self.class_num, kernel_initializer="he_normal", use_bias=False, activation="softmax", name="identifier")(base_model.output)
            self.model = Model(inputs=base_model.input, outputs=identifiers)

        # self.model.summary()

    def get_optimizer(self):
        if self.optimizer == 'sgd':
            return SGD(learning_rate=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        elif self.optimizer == 'adam':
            return Adam(learning_rate=self.learning_rate)
        elif self.optimizer == 'nadam':
            return Nadam(learning_rate=self.learning_rate)

    def read_total_epochs(self, file_path):
        if os.path.exists(file_path):
            with open(file_path) as f:
                try:
                    context = json.loads(f.read())
                except:
                    return 0
                return context['epochs']
        else:
            return 0

    def write_total_epochs(self, file_path):
        with open(file_path, 'w') as f:
            f.write(json.dumps({'epochs':self.total_epochs}))

    def train(self):
        # faces, labels = self.get_datasets()
        def capsule_loss(y_true, y_pred):
            return y_true*K.relu(0.9-y_pred)**2 + 0.25*(1-y_true)*K.relu(y_pred-0.1)**2

        if not self.ckpt_exist:
            self.log_info('start to train')
            opt = self.get_optimizer()
            if self.network not in ['CustomInceptionResNetV2']:
                self.model.compile(loss='categorical_crossentropy',
                                   optimizer=opt,
                                   metrics=['accuracy'])
            else:
                self.model.compile(loss=capsule_loss,
                                   optimizer=opt,
                                   metrics=['accuracy'])
        else:
            self.log_info('loading checkpoint')
            if self.network not in ['CustomInceptionResNetV2']:
                self.model = load_model(os.path.join(self.checkpoint, 'model.h5'))
            else:
                self.model = load_model(os.path.join(self.checkpoint, 'model.h5'),
                                        custom_objects={'CustomInceptionResNetV2': CustomInceptionResNetV2,
                                                        'capsule_loss': capsule_loss})

        train_datagen = ImageDataGenerator(rescale = 1./255,
                                           shear_range = 0.2,
                                           zoom_range = 0.2,
                                           rotation_range=30,
                                           horizontal_flip=True)
        val_datagen = ImageDataGenerator(rescale = 1./255,
                                 shear_range = 0.2,
                                 zoom_range = 0.2,
                                 rotation_range=30,
                                 horizontal_flip=True)

        eval_datagen = ImageDataGenerator(rescale = 1./255,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  rotation_range=30,
                                  horizontal_flip=True)

        train_generator = train_datagen.flow_from_directory(
                                    os.path.join(self.dataset, 'train'),
                                    target_size=(self.image_size, self.image_size),
                                    # color_mode='grayscale',
                                    batch_size=self.batch_size,
                                    class_mode='categorical')
        val_generator = val_datagen.flow_from_directory(
                                    os.path.join(self.dataset, 'val'),
                                    target_size=(self.image_size, self.image_size),
                                    # color_mode='grayscale',
                                    batch_size=self.batch_size,
                                    class_mode='categorical')
        eval_generator = eval_datagen.flow_from_directory(
                                    os.path.join(self.dataset, 'test'),
                                    target_size=(self.image_size, self.image_size),
                                    # color_mode='grayscale',
                                    batch_size=self.batch_size,
                                    class_mode='categorical')
        num_images = len(eval_generator.filenames)
        num_steps = max(1, num_images // self.batch_size)
        self.total_epochs = self.read_total_epochs(os.path.join(self.checkpoint, 'train.json'))
        # label_encoder = LabelEncoder()
        # integer_encoded = label_encoder.fit_transform(labels)

        # One-hot encode the integer labels
        # onehot_encoded = to_categorical(integer_encoded, num_classes=self.class_num)
        history_fit = None
        interrupted = False
        self.log_info('start epoch : {}'.format(self.total_epochs))
        try:
            history_fit = self.model.fit(train_generator,
                        epochs=self.epochs,
                        validation_data=val_generator,
                        shuffle=True,
                        verbose=1)
        except KeyboardInterrupt:
            self.log_info('\n\nKeyboard interrupt!')
            interrupted = True

        if not interrupted:
            self.total_epochs += self.epochs

        history_predict=self.model.predict(
                eval_generator,
                steps=num_steps)
        try:
            if history_fit:
                with open(os.path.join(self.checkpoint, 'model_fit_log'),'w') as f:
                    f.write(str(history_fit.history))
            if history_predict is not None:
                with open(os.path.join(self.checkpoint, 'model_predict_log'),'w') as f:
                    f.write(str(history_predict))
            self.write_total_epochs(os.path.join(self.checkpoint, 'train.json'))
        except Exception as e:
            import traceback
            print(traceback.format_exc())

    def save_model(self):
        model_json = self.model.to_json()
        with open(os.path.join(self.checkpoint, "model_json.json"), "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(os.path.join(self.checkpoint, 'model_weight.h5'))
        self.model.save(os.path.join(self.checkpoint, 'model.h5'))
        print('model saved')


def get_args():
    parser = argparse.ArgumentParser(description="Train for emotion prediction.")
    parser.add_argument('--dataset', '-d',
                        help="path to dataset.", default='datasets/')
    parser.add_argument('--checkpoint', '-c', default='ckpt/',
                        help="checkpoint directory.")
    parser.add_argument('--batch-size', '-b', type=int, default=64,
                        help="batch size")
    parser.add_argument('--epochs', '-e', type=int, default=70,
                        help="number of epochs")
    parser.add_argument('--lr', '-l', type=float, default=0.0001,
                        help="learning rate")
    parser.add_argument('--opt', default='adam', choices=['sgd', 'adam', 'nadam'],
                        help="optimizer: sgd or adam")
    parser.add_argument('--network', '-n', default='InceptionResNetV2', choices=networks,
                        help="network: {}".format(', '.join(networks)))
    parser.add_argument('--image-size', '-s', type=int, default=299,
                        help="image width or height")
    parser.add_argument('--channel', type=int, default=3,
                        help="image channel number")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    e = FaceRecognition(args)
    e.build_model()
    e.train()
    e.save_model()

if __name__ == '__main__':
    main()
