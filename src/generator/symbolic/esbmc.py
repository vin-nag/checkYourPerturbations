from .symbolicExecutor import SymbolicExecutioner
import pdb

class ESBMC(SymbolicExecutioner):
    def __init__(self, name, model, image, label, similarityType="l2", similarityMeasure=10):
        super().__init__(name, model, image, label, similarityType, similarityMeasure)

    def mk_esmbc_file(self,file):
        n_features = self.weights[0]
        with open(file, "w") as f:
            f.write("#include <math.h>\n")
            f.write("int main() {\n")
        for i in range(len(n_features)):
            with open(file, "a") as f:
                f.write(f"float feature_{i} = nondet_float(); __VERIFIER_assume(feature_{i} <= 1.0 && feature_{i} >= -1.0);\n")

        # RELU fully connected layers
        for x in range(len(self.weights) - 1):
            for i, vals in enumerate(self.weights[x]):
                line = f"float prelayer{x + 1}_{i} = "
                for j, val in enumerate(vals):
                    if x != 0: 
                        line = line + f"{val}*prelayer{x}_{j} + "
                        line = line + f"{self.biases[x][i]})"
                    else: line = line + f"{val}*feature_{j};\n"
                with open(file, "a") as f:
                    f.write(f"{line};\n")
                with open(file, "a") as f:
                    line = f"float layer{x + 1}_{i} = (prelayer{x + 1}_{i} + abs(prelayer{x + 1}_{i})) / 2"
                    f.write(f"{line};\n")

        # Exponents for softmax final layer

        for i, vals in enumerate(self.weights[-1]):
            line = f"float layer{len(self.weights)}_{i} = exp("
            for j, val in enumerate(vals):
                line = line + f"{val}*layer{len(self.weights) - 1}_{j} + "
                line = line + f"{self.biases[len(self.weights) - 1][j]})"
            # print(line)
            with open(file, "a") as f:
                f.write(f"{line};\n")

        # Exponent sum for softmax final layer
        line = "float expsum ="
        for i in range(len(self.weights[-1])):
            line = line + f" layer{len(self.weights)}_{i} + "
        line = line[:-2]
        # print(line)
        with open(file, "a") as f:
            f.write(f"{line};\n")

        # Softmax calculation
        for i in range(len(self.weights[-1])):
            line = f"float logit{i} = layer{len(self.weights)}_{i} / expsum"
            # print(line)
            with open(file, "a") as f:
                f.write(f"{line};\n")

        with open(file, "a") as f:
            f.write("assert(logit0 > logit1);")
            f.write("return 0;\n}")
    
    def solve(self):
        pdb.set_trace()
        self.mk_esmbc_file("tmp.c")