"""
This file contains the DLFUzz code (from: https://github.com/turned2670/DLFuzz) and modified by Laura

Date:
    July 26, 2020

Project:
    ECE653 Final Project: Check Your Perturbations

Authors:
    name: Vineel Nagisetty, Laura Graves, Joseph Scott
    contact: vineel.nagisetty@uwaterloo.ca
"""

from src.generator.template import GeneratorTemplate
from collections import defaultdict
from keras import backend as k
from tensorflow.keras.models import Model
import numpy as np
import time


class DLFuzzer(GeneratorTemplate):

    def __init__(self, name, model, modelName, image, label, similarityType="l2", similarityMeasure=10, verbose=True):
        super().__init__(name, model, modelName, image, label, similarityType, similarityMeasure, verbose)
        self.model_layer_weights_top_k = []

    @staticmethod
    def normalize(x):
        # utility function to normalize a tensor by its L2 norm
        return x / (k.sqrt(k.mean(k.square(x))) + 1e-5)

    def init_coverage_times(self):
        model_layer_times = defaultdict(int)
        self.init_times(model_layer_times)
        return model_layer_times

    def init_coverage_value(self):
        model_layer_value = defaultdict(float)
        self.init_times(model_layer_value)
        return model_layer_value

    def init_times(self, model_layer_times):
        for layer in self.model.layers:
            if 'flatten' in layer.name or 'input' in layer.name:
                continue
            for index in range(layer.output_shape[-1]):
                model_layer_times[(layer.name, index)] = 0

    def neuron_select_high_weight(self, layer_names, top_k):
        model_layer_weights_dict = {}
        for layer_name in layer_names:
            weights = self.model.get_layer(layer_name).get_weights()
            if len(weights) <= 0:
                continue
            w = np.asarray(weights[0])  # 0 is weights, 1 is biases
            w = w.reshape(w.shape)
            for index in range(self.model.get_layer(layer_name).output_shape[-1]):
                index_w = np.mean(w[..., index])
                if index_w <= 0:
                    continue
                model_layer_weights_dict[(layer_name, index)] = index_w
        model_layer_weights_list = sorted(model_layer_weights_dict.items(), key=lambda x: x[1], reverse=True)
        j = 0
        for (layer_name, index), weight in model_layer_weights_list:
            if j >= top_k:
                break
            self.model_layer_weights_top_k.append([layer_name, index])
            j += 1

    def neuron_selection(self, model_layer_times, neuron_to_cover_num):
        loss_neuron = []
        # select neurons with largest weights (feature maps with largest filter weights)
        layer_names = [layer.name for layer in self.model.layers if
                       'flatten' not in layer.name and 'input' not in layer.name]
        j = 0.1
        top_k = j * len(model_layer_times)  # number of neurons to be selected within
        if len(self.model_layer_weights_top_k) == 0:
            self.neuron_select_high_weight(layer_names, top_k)  # Set the value
        num_neuron2 = np.random.choice(range(len(self.model_layer_weights_top_k)), neuron_to_cover_num, replace=False)
        for i in num_neuron2:
            layer_name2 = self.model_layer_weights_top_k[i][0]
            index2 = self.model_layer_weights_top_k[i][1]
            loss2_neuron = k.mean(self.model.get_layer(layer_name2).output[..., index2])
            loss_neuron.append(loss2_neuron)
        return loss_neuron

    @staticmethod
    def neuron_covered(model_layer_times):
        covered_neurons = len([v for v in model_layer_times.values() if v > 0])
        total_neurons = len(model_layer_times)
        return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

    @staticmethod
    def scale(intermediate_layer_output, rmax=1, rmin=0):
        X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
                intermediate_layer_output.max() - intermediate_layer_output.min())
        X_scaled = X_std * (rmax - rmin) + rmin
        return X_scaled

    def update_coverage(self, input_data, model_layer_times, threshold=0.0):
        layer_names = [layer.name for layer in self.model.layers if
                       'flatten' not in layer.name and 'input' not in layer.name]
        intermediate_layer_model = Model(inputs=self.model.input,
                                         outputs=[self.model.get_layer(name).output for name in layer_names])
        intermediate_layer_outputs = intermediate_layer_model.predict(input_data)
        for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
            scaled = self.scale(intermediate_layer_output[0])
            for num_neuron in range(scaled.shape[-1]):
                if np.mean(scaled[..., num_neuron]) > threshold:
                    model_layer_times[(layer_names[i], num_neuron)] += 1
        return intermediate_layer_outputs

    def update_coverage_value(self, input_data, model_layer_value):
        layer_names = [layer.name for layer in self.model.layers if
                       'flatten' not in layer.name and 'input' not in layer.name]
        intermediate_layer_model = Model(inputs=self.model.input,
                                         outputs=[self.model.get_layer(name).output for name in layer_names])
        intermediate_layer_outputs = intermediate_layer_model.predict(input_data)
        for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
            scaled = self.scale(intermediate_layer_output[0])
            for num_neuron in range(scaled.shape[-1]):
                model_layer_value[(layer_names[i], num_neuron)] = np.mean(scaled[..., num_neuron])
        return intermediate_layer_outputs

    def generateAdversarialExample(self):
        start_time = time.time()
        model_layer_times1 = self.init_coverage_times()  # times of each neuron covered
        model_layer_times2 = self.init_coverage_times()  # update when new image and adversarial images found
        model_layer_value1 = self.init_coverage_value()

        threshold = 0.25
        neuron_to_cover_num = 10
        iteration_times = 0

        neuron_to_cover_weight = 0.5
        predict_weight = 0.5
        learning_step = 0.02
        total_norm = 0

        img_list = []
        adv_list = []
        adv_labels = []

        while len(adv_list) < 1:

            iteration_times += 1

            total_perturb_adversarial = 0

            tmp_img = self.image.reshape(1, 28, 28, 1)
            orig_img = tmp_img.copy()

            img_list.append(tmp_img)

            self.update_coverage(tmp_img, model_layer_times2, threshold)

            while len(img_list) > 0:

                gen_img = img_list[0]

                img_list.remove(gen_img)

                # first check if input already induces differences
                pred1 = self.model.predict(gen_img)
                label1 = np.argmax(pred1[0])

                label_top5 = np.argsort(pred1[0])[-5:]

                self.update_coverage_value(gen_img, model_layer_value1)
                self.update_coverage(gen_img, model_layer_times1, threshold)

                orig_label = label1

                loss_1 = k.mean(self.model.get_layer('before_softmax').output[..., orig_label])
                loss_2 = k.mean(self.model.get_layer('before_softmax').output[..., label_top5[-2]])
                loss_3 = k.mean(self.model.get_layer('before_softmax').output[..., label_top5[-3]])
                loss_4 = k.mean(self.model.get_layer('before_softmax').output[..., label_top5[-4]])
                loss_5 = k.mean(self.model.get_layer('before_softmax').output[..., label_top5[-5]])

                layer_output = (predict_weight * (loss_2 + loss_3 + loss_4 + loss_5) - loss_1)

                # neuron coverage loss
                loss_neuron = self.neuron_selection(model_layer_times1, neuron_to_cover_num)

                # extreme value means the activation value for a neuron can be as high as possible ...

                layer_output += neuron_to_cover_weight * k.sum(loss_neuron)

                # for adversarial image generation
                final_loss = k.mean(layer_output)

                # we compute the gradient of the input picture wrt this loss
                grads = self.normalize(k.gradients(final_loss, self.model.input)[0])

                grads_tensor_list = [loss_1, loss_2, loss_3, loss_4, loss_5]
                grads_tensor_list.extend(loss_neuron)
                grads_tensor_list.append(grads)
                # this function returns the loss and grads given the input picture

                iterate = k.function([self.model.input], grads_tensor_list)

                # we run gradient ascent for 3 steps
                for iters in range(iteration_times):

                    loss_neuron_list = iterate([gen_img])

                    perturb = loss_neuron_list[-1] * learning_step

                    gen_img += perturb
                    gen_img = np.clip(gen_img, -0.5, 5)

                    # previous accumulated neuron coverage
                    previous_coverage = self.neuron_covered(model_layer_times1)[2]

                    pred1 = self.model.predict(gen_img)
                    label1 = np.argmax(pred1[0])

                    self.update_coverage(gen_img, model_layer_times1, threshold)  # for seed selection

                    current_coverage = self.neuron_covered(model_layer_times1)[2]

                    diff_img = gen_img - orig_img

                    L2_norm = np.linalg.norm(diff_img)

                    orig_L2_norm = np.linalg.norm(orig_img)

                    perturb_adversial = L2_norm / orig_L2_norm

                    if current_coverage - previous_coverage > 0.0001 and L2_norm < self.similarityMeasure:
                        img_list.append(np.clip(gen_img, -0.5, 5))

                    if label1 != orig_label and L2_norm < self.similarityMeasure:
                        self.update_coverage(gen_img, model_layer_times2, threshold)

                        total_norm += L2_norm

                        total_perturb_adversarial += perturb_adversial

                        gen_img = np.clip(gen_img, -0.5, 5)

                        adv_list.append(gen_img)
                        adv_labels.append(np.argmax(self.model.predict(gen_img)[0]))

        end_time = time.time()
        self.time = end_time - start_time
        self.advLabel = adv_labels[0]
        self.advImage = adv_list[0]
        self.completed = True

        if self.verbose:
            print('\ncovered neurons percentage %d neurons %.3f'
                  % (len(model_layer_times2), self.neuron_covered(model_layer_times2)[2]))
