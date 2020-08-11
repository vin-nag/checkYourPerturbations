from .symbolicExecutor import SymbolicExecutioner
import pdb

class ESBMC(SymbolicExecutioner):
    def __init__(self, name, model, modelName, image, label, similarityType="l2", similarityMeasure=10):
        super().__init__(name, model, modelName, image, label, similarityType, similarityMeasure)

    def mk_esmbc_file(self,file):
        n_features = self.weights[0].shape[1]
        with open(file, "w") as f:
            f.write("#include <math.h>\n")
            f.write("int main() {\n")
        flat_image = self.image.flatten()
        with open(file, "a") as f:
            for i in range(n_features):
                f.write(f"float feature_{i} = nondet_float(); __VERIFIER_assume(feature_{i} <= 0.5f && feature_{i} >= -0.5f && fabsf(feature_{i} - {flat_image[i]}) < 0.01f);\n")
        for it_layer in range(len(self.weights) - 1):
            for it, weights in enumerate(self.weights[it_layer]):
                line = f"float neuron{it_layer}_{it} = "
                for jt, val in enumerate(weights):
                    if it_layer == 0: 
                        line += f"{val}f * feature_{jt} +"
                    else:
                        line += f"{val}f* layer{it_layer}_{it_weights} + "
                line += f"{self.biases[it_layer][it]};\n"
                with open(file, "a") as f:
                    f.write(line)
                    f.write(f"float layer{it_layer}_{it} = (neuron{it_layer}_{it} + fabsf(neuron{it_layer}_{it})) / 2.0f;\n")

        # Output Layer
        for it, vals in enumerate(self.weights[-1]):
            line = f"float output_{it} = "
            for jt, val in enumerate(vals):
                line += f"{val}*layer{len(self.weights) - 2}_{jt} + "
            line += f"{self.biases[len(self.weights) - 1][it]}"
            with open(file, "a") as f:
                f.write(f"{line};\n")

        # # Exponent sum for softmax final layer
        # line = "float expsum ="
        # for i in range(len(self.weights[-1])):
        #     line = line + f" layer{len(self.weights)}_{i} + "
        # line = line[:-2]
        # with open(file, "a") as f:
        #     f.write(f"{line};\n")

        # # Softmax calculation
        # for i in range(len(self.weights[-1])):
        #     line = f"float logit{i} = layer{len(self.weights)}_{i} / expsum"
        #     # print(line)
        #     with open(file, "a") as f:
        #         f.write(f"{line};\n")

        pdb.set_trace()
        with open(file, "a") as f:
            line = 'assert('
            for i in range(len(self.weights[-1])):
                if i == self.label: continue
                line += f'output_{i} > output_{self.label} ||'
            line = line[:-2] + ');\n'
            f.write(line)
            f.write("return 0;\n}")
    
    def solve(self):
        self.mk_esmbc_file("tmp.c")