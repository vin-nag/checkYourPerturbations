from .symbolicExecutor import SymbolicExecutioner
import pdb
from z3 import FPSort, FPVal, fpAbs, Solver, FP, fpLEQ, fpGEQ, fpLT, fpGT, fpSub, fpAdd, fpDiv, fpMul, fpEQ, Or, RNE

class SMT(SymbolicExecutioner):
    def __init__(self, name, model, modelName, image, label, similarityType="l2", similarityMeasure=10, eb=5, sb=11, upper_bound=1, lower_bound=-1, use_float = False, use_real = False):
        super().__init__(name, model, modelName, image, label, similarityType, similarityMeasure)
        self.float, self.real = use_float, use_real 
        self.eb, self.sb = eb, sb
        self.upper_bound, self.lower_bound = upper_bound, lower_bound
        if not(use_float or use_real): self.float = True #default to fp 
        self.solver = Solver()

    def mk_smt(self):
        sort = FPSort(self.eb,self.sb)
        n_features = self.weights[0].shape[1]
        pdb.set_trace()
        flat_image = self.image.numpy().flatten()
        features = [FP(f'feature_{i}', sort) for i in range(n_features)]
        for it, f in enumerate(features):
            self.solver.add(fpLEQ(f, FPVal(self.upper_bound, sort))) 
            self.solver.add(fpGEQ(f, FPVal(self.lower_bound, sort)))
            self.solver.add(
                fpLT(
                    fpAbs(fpSub(RNE(), f, FPVal(float(flat_image[it]),sort))),
                    FPVal(float(self.similarityMeasure),sort)
                )
            )
        neurons = {}
        layers = {}
        for it_layer in range(len(self.weights) - 1):
            for it, weights in enumerate(self.weights[it_layer]):
                if it_layer not in neurons: 
                    neurons[it_layer] = {}
                    layers[it_layer]  = {}
                neurons[it_layer][it] = FP(f'neuron_{it_layer}_{it}', sort)
                layers[it_layer][it]  = FP(f'layer_{it_layer}_{it}' , sort)
                neuron = None
                for jt, val in enumerate(weights):
                    w = FPVal(float(val), sort)
                    x = features[jt] if it_layer == 0 else layers[it_layer]
                    neuron = fpMul(RNE(), w, x) if neuron == None else fpAdd(RNE(), neuron, fpMul(RNE(), w, x))
                neuron = fpAdd(RNE(), neuron, FPVal(float(self.biases[it_layer][it]), sort))
                self.solver.add(fpEQ(neuron, neurons[it_layer][it]))
                self.solver.add(fpEQ(
                    fpDiv(RNE(), 
                        fpAdd(RNE(), neurons[it_layer][it], fpAbs(neurons[it_layer][it])),
                        FPVal(2.0,sort)),
                    layers[it_layer][it]
                ))
        # Exponents for softmax final layer
        outputs = {}
        for it, vals in enumerate(self.weights[-1]):
            outputs[it] = FP(f'output_{it}', sort)
            out = None
            for jt, val in enumerate(vals):
                w = FPVal(float(val), sort)
                x = layers[len(self.weights) -2][jt]
                out = fpMul(RNE(), w, x) if out == None else fpAdd(RNE(), out, fpMul(RNE(), w, x))
            out = fpAdd(RNE(), out, FPVal(float(self.biases[len(self.weights) - 1][it]),sort))
            self.solver.add(fpEQ(
                outputs[it],
                out
            ))
            check = None
            check = None
        check = None
        for it in range(len(self.weights[-1])):
            if it == self.label: continue
            check = fpGT(outputs[it], outputs[self.label]) if check == None else Or(check, fpGT(outputs[it], outputs[self.label]))
        self.solver.add(check)
    def solve(self):
        self.mk_smt()
        with open('tmp.smt2', 'w') as f:
            f.write(self.solver.to_smt2())