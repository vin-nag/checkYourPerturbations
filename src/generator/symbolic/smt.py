from .symbolicExecutor import SymbolicExecutioner
import pdb
from z3 import FPSort, FPVal, fpAbs, Solver, FP, fpLEQ, fpGEQ, fpLT, fpGT, fpSub, RNE

class SMT(SymbolicExecutioner):
    def __init__(self, name, model, modelName, image, label, similarityType="l2", similarityMeasure=10):
        super().__init__(name, model, modelName, image, label, similarityType, similarityMeasure)
        self.float, self.real = use_float,use_real 
        self.eb, self.sb = eb, sb
        self.upper_bound, self.lower_bound = upper_bound, lower_bound
        if not(use_float or use_real): self.float = True #default to fp 
        self.solver = Solver()

    def mk_smt(self):
        sort = FPSort(self.eb,self.sb)
        n_features = self.weights[0].shape[1]
        flat_image = self.image.numpy().flatten()
        features = [FP(f'feature_{i}', sort) for i in range(n_features)]
        for it, f in enumerate(features):
            self.solver.check(fpLEQ(f, FPVal(self.upper_bound, sort))) 
            self.solver.check(fpGEQ(f, FPVal(self.lower_bound, sort)))
            self.solver.add(
                fpLT(
                    fpAbs(fpSub(RNE(), f, FPVal(float(flat_image[it]),sort))),
                    FPVal(float(self.similarityMeasure),sort)
                )
            )
        nerouns = {}
        layers = {}
        for it_layer in range(len(self.weights) - 1):
            for it, weights in enumerate(self.weights[it_layer]):
                if it_layer not in neurons: 
                    neurons[it_layer] = {}
                    layers[it_layer]  = {}
                neurons[it_layer][it] = FP(f'neuron_{it_layer}_{it}', sort)
                for jt, val in enumerate(weights):
                    w = features[jt] if it_layer == 0 else layers[it_layer]
                    if it_layer == 0: 
                        neurons[it_layer][jt]
                        line += f"{val}f * feature_{jt} +"
                    else:
                        line += f"{val}f* layer{it_layer}_{it_weights} + "
                line += f"{self.biases[it_layer][it]};\n"
                with open(file, "a") as f:
                    f.write(line)
                    f.write(f"float layer{it_layer}_{it} = (neuron{it_layer}_{it} + fabsf(neuron{it_layer}_{it})) / 2.0f;\n")

        # Exponents for softmax final layer
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
        self.mk_smt()