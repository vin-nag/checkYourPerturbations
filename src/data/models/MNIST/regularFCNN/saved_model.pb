˙é
Şý
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
ž
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8é

before_softmax_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_namebefore_softmax_3/kernel

+before_softmax_3/kernel/Read/ReadVariableOpReadVariableOpbefore_softmax_3/kernel* 
_output_shapes
:
*
dtype0

before_softmax_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namebefore_softmax_3/bias
|
)before_softmax_3/bias/Read/ReadVariableOpReadVariableOpbefore_softmax_3/bias*
_output_shapes	
:*
dtype0

after_softmax_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*'
shared_nameafter_softmax_3/kernel

*after_softmax_3/kernel/Read/ReadVariableOpReadVariableOpafter_softmax_3/kernel*
_output_shapes
:	
*
dtype0

after_softmax_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameafter_softmax_3/bias
y
(after_softmax_3/bias/Read/ReadVariableOpReadVariableOpafter_softmax_3/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/before_softmax_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*/
shared_name Adam/before_softmax_3/kernel/m

2Adam/before_softmax_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/before_softmax_3/kernel/m* 
_output_shapes
:
*
dtype0

Adam/before_softmax_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/before_softmax_3/bias/m

0Adam/before_softmax_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/before_softmax_3/bias/m*
_output_shapes	
:*
dtype0

Adam/after_softmax_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*.
shared_nameAdam/after_softmax_3/kernel/m

1Adam/after_softmax_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/after_softmax_3/kernel/m*
_output_shapes
:	
*
dtype0

Adam/after_softmax_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_nameAdam/after_softmax_3/bias/m

/Adam/after_softmax_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/after_softmax_3/bias/m*
_output_shapes
:
*
dtype0

Adam/before_softmax_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*/
shared_name Adam/before_softmax_3/kernel/v

2Adam/before_softmax_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/before_softmax_3/kernel/v* 
_output_shapes
:
*
dtype0

Adam/before_softmax_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/before_softmax_3/bias/v

0Adam/before_softmax_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/before_softmax_3/bias/v*
_output_shapes	
:*
dtype0

Adam/after_softmax_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*.
shared_nameAdam/after_softmax_3/kernel/v

1Adam/after_softmax_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/after_softmax_3/kernel/v*
_output_shapes
:	
*
dtype0

Adam/after_softmax_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_nameAdam/after_softmax_3/bias/v

/Adam/after_softmax_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/after_softmax_3/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
Ę
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueűBř Bń
Ě
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api
	
signatures
R

	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api

iter

beta_1

beta_2
	decay
learning_ratem>m?m@mAvBvCvDvE

0
1
2
3
 

0
1
2
3
­
	variables
non_trainable_variables

 layers
!metrics
"layer_metrics
regularization_losses
trainable_variables
#layer_regularization_losses
 
 
 
 
­

	variables
$non_trainable_variables

%layers
&metrics
'layer_metrics
regularization_losses
trainable_variables
(layer_regularization_losses
ca
VARIABLE_VALUEbefore_softmax_3/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEbefore_softmax_3/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
	variables
)non_trainable_variables

*layers
+metrics
,layer_metrics
regularization_losses
trainable_variables
-layer_regularization_losses
b`
VARIABLE_VALUEafter_softmax_3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEafter_softmax_3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
	variables
.non_trainable_variables

/layers
0metrics
1layer_metrics
regularization_losses
trainable_variables
2layer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2

30
41
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	5total
	6count
7	variables
8	keras_api
D
	9total
	:count
;
_fn_kwargs
<	variables
=	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

50
61

7	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

90
:1

<	variables

VARIABLE_VALUEAdam/before_softmax_3/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/before_softmax_3/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/after_softmax_3/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/after_softmax_3/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/before_softmax_3/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/before_softmax_3/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/after_softmax_3/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/after_softmax_3/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_flatten_3_inputPlaceholder*+
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0* 
shape:˙˙˙˙˙˙˙˙˙
ţ
StatefulPartitionedCallStatefulPartitionedCallserving_default_flatten_3_inputbefore_softmax_3/kernelbefore_softmax_3/biasafter_softmax_3/kernelafter_softmax_3/bias*
Tin	
2*
Tout
2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*-
f(R&
$__inference_signature_wrapper_115942
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
é
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+before_softmax_3/kernel/Read/ReadVariableOp)before_softmax_3/bias/Read/ReadVariableOp*after_softmax_3/kernel/Read/ReadVariableOp(after_softmax_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp2Adam/before_softmax_3/kernel/m/Read/ReadVariableOp0Adam/before_softmax_3/bias/m/Read/ReadVariableOp1Adam/after_softmax_3/kernel/m/Read/ReadVariableOp/Adam/after_softmax_3/bias/m/Read/ReadVariableOp2Adam/before_softmax_3/kernel/v/Read/ReadVariableOp0Adam/before_softmax_3/bias/v/Read/ReadVariableOp1Adam/after_softmax_3/kernel/v/Read/ReadVariableOp/Adam/after_softmax_3/bias/v/Read/ReadVariableOpConst*"
Tin
2	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*(
f#R!
__inference__traced_save_116149
Ŕ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebefore_softmax_3/kernelbefore_softmax_3/biasafter_softmax_3/kernelafter_softmax_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/before_softmax_3/kernel/mAdam/before_softmax_3/bias/mAdam/after_softmax_3/kernel/mAdam/after_softmax_3/bias/mAdam/before_softmax_3/kernel/vAdam/before_softmax_3/bias/vAdam/after_softmax_3/kernel/vAdam/after_softmax_3/bias/v*!
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*+
f&R$
"__inference__traced_restore_116224

 
-__inference_sequential_3_layer_call_fn_115995

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCallđ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_1158802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

Ť
H__inference_sequential_3_layer_call_and_return_conditional_losses_115847
flatten_3_input
before_softmax_115814
before_softmax_115816
after_softmax_115841
after_softmax_115843
identity˘%after_softmax/StatefulPartitionedCall˘&before_softmax/StatefulPartitionedCallż
flatten_3/PartitionedCallPartitionedCallflatten_3_input*
Tin
2*
Tout
2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1157842
flatten_3/PartitionedCall­
&before_softmax/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0before_softmax_115814before_softmax_115816*
Tin
2*
Tout
2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_before_softmax_layer_call_and_return_conditional_losses_1158032(
&before_softmax/StatefulPartitionedCall´
%after_softmax/StatefulPartitionedCallStatefulPartitionedCall/before_softmax/StatefulPartitionedCall:output:0after_softmax_115841after_softmax_115843*
Tin
2*
Tout
2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_after_softmax_layer_call_and_return_conditional_losses_1158302'
%after_softmax/StatefulPartitionedCallÓ
IdentityIdentity.after_softmax/StatefulPartitionedCall:output:0&^after_softmax/StatefulPartitionedCall'^before_softmax/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙::::2N
%after_softmax/StatefulPartitionedCall%after_softmax/StatefulPartitionedCall2P
&before_softmax/StatefulPartitionedCall&before_softmax/StatefulPartitionedCall:\ X
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameflatten_3_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


.__inference_after_softmax_layer_call_fn_116059

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_after_softmax_layer_call_and_return_conditional_losses_1158302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ü
F
*__inference_flatten_3_layer_call_fn_116019

inputs
identity˘
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1157842
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0**
_input_shapes
:˙˙˙˙˙˙˙˙˙:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ô
˛
J__inference_before_softmax_layer_call_and_return_conditional_losses_115803

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ĺ
˘
H__inference_sequential_3_layer_call_and_return_conditional_losses_115880

inputs
before_softmax_115869
before_softmax_115871
after_softmax_115874
after_softmax_115876
identity˘%after_softmax/StatefulPartitionedCall˘&before_softmax/StatefulPartitionedCallś
flatten_3/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1157842
flatten_3/PartitionedCall­
&before_softmax/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0before_softmax_115869before_softmax_115871*
Tin
2*
Tout
2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_before_softmax_layer_call_and_return_conditional_losses_1158032(
&before_softmax/StatefulPartitionedCall´
%after_softmax/StatefulPartitionedCallStatefulPartitionedCall/before_softmax/StatefulPartitionedCall:output:0after_softmax_115874after_softmax_115876*
Tin
2*
Tout
2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_after_softmax_layer_call_and_return_conditional_losses_1158302'
%after_softmax/StatefulPartitionedCallÓ
IdentityIdentity.after_softmax/StatefulPartitionedCall:output:0&^after_softmax/StatefulPartitionedCall'^before_softmax/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙::::2N
%after_softmax/StatefulPartitionedCall%after_softmax/StatefulPartitionedCall2P
&before_softmax/StatefulPartitionedCall&before_softmax/StatefulPartitionedCall:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ĺ
˘
H__inference_sequential_3_layer_call_and_return_conditional_losses_115908

inputs
before_softmax_115897
before_softmax_115899
after_softmax_115902
after_softmax_115904
identity˘%after_softmax/StatefulPartitionedCall˘&before_softmax/StatefulPartitionedCallś
flatten_3/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1157842
flatten_3/PartitionedCall­
&before_softmax/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0before_softmax_115897before_softmax_115899*
Tin
2*
Tout
2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_before_softmax_layer_call_and_return_conditional_losses_1158032(
&before_softmax/StatefulPartitionedCall´
%after_softmax/StatefulPartitionedCallStatefulPartitionedCall/before_softmax/StatefulPartitionedCall:output:0after_softmax_115902after_softmax_115904*
Tin
2*
Tout
2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_after_softmax_layer_call_and_return_conditional_losses_1158302'
%after_softmax/StatefulPartitionedCallÓ
IdentityIdentity.after_softmax/StatefulPartitionedCall:output:0&^after_softmax/StatefulPartitionedCall'^before_softmax/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙::::2N
%after_softmax/StatefulPartitionedCall%after_softmax/StatefulPartitionedCall2P
&before_softmax/StatefulPartitionedCall&before_softmax/StatefulPartitionedCall:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ľ
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_116014

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0**
_input_shapes
:˙˙˙˙˙˙˙˙˙:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ű;
Ä	
__inference__traced_save_116149
file_prefix6
2savev2_before_softmax_3_kernel_read_readvariableop4
0savev2_before_softmax_3_bias_read_readvariableop5
1savev2_after_softmax_3_kernel_read_readvariableop3
/savev2_after_softmax_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop=
9savev2_adam_before_softmax_3_kernel_m_read_readvariableop;
7savev2_adam_before_softmax_3_bias_m_read_readvariableop<
8savev2_adam_after_softmax_3_kernel_m_read_readvariableop:
6savev2_adam_after_softmax_3_bias_m_read_readvariableop=
9savev2_adam_before_softmax_3_kernel_v_read_readvariableop;
7savev2_adam_before_softmax_3_bias_v_read_readvariableop<
8savev2_adam_after_softmax_3_kernel_v_read_readvariableop:
6savev2_adam_after_softmax_3_bias_v_read_readvariableop
savev2_1_const

identity_1˘MergeV2Checkpoints˘SaveV2˘SaveV2_1
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_d85a52d961e246018168a122f3766b24/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardŚ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename´
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ć

valueź
Bš
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names˛
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesŽ	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_before_softmax_3_kernel_read_readvariableop0savev2_before_softmax_3_bias_read_readvariableop1savev2_after_softmax_3_kernel_read_readvariableop/savev2_after_softmax_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop9savev2_adam_before_softmax_3_kernel_m_read_readvariableop7savev2_adam_before_softmax_3_bias_m_read_readvariableop8savev2_adam_after_softmax_3_kernel_m_read_readvariableop6savev2_adam_after_softmax_3_bias_m_read_readvariableop9savev2_adam_before_softmax_3_kernel_v_read_readvariableop7savev2_adam_before_softmax_3_bias_v_read_readvariableop8savev2_adam_after_softmax_3_kernel_v_read_readvariableop6savev2_adam_after_softmax_3_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *#
dtypes
2	2
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardŹ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1˘
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesĎ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1ă
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesŹ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: :
::	
:
: : : : : : : : : :
::	
:
:
::	
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	
: 

_output_shapes
:
:&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	
: 

_output_shapes
:
:

_output_shapes
: 

Š
-__inference_sequential_3_layer_call_fn_115891
flatten_3_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCallů
StatefulPartitionedCallStatefulPartitionedCallflatten_3_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_1158802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameflatten_3_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
`
Č
"__inference__traced_restore_116224
file_prefix,
(assignvariableop_before_softmax_3_kernel,
(assignvariableop_1_before_softmax_3_bias-
)assignvariableop_2_after_softmax_3_kernel+
'assignvariableop_3_after_softmax_3_bias 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rate
assignvariableop_9_total
assignvariableop_10_count
assignvariableop_11_total_1
assignvariableop_12_count_16
2assignvariableop_13_adam_before_softmax_3_kernel_m4
0assignvariableop_14_adam_before_softmax_3_bias_m5
1assignvariableop_15_adam_after_softmax_3_kernel_m3
/assignvariableop_16_adam_after_softmax_3_bias_m6
2assignvariableop_17_adam_before_softmax_3_kernel_v4
0assignvariableop_18_adam_before_softmax_3_bias_v5
1assignvariableop_19_adam_after_softmax_3_kernel_v3
/assignvariableop_20_adam_after_softmax_3_bias_v
identity_22˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_10˘AssignVariableOp_11˘AssignVariableOp_12˘AssignVariableOp_13˘AssignVariableOp_14˘AssignVariableOp_15˘AssignVariableOp_16˘AssignVariableOp_17˘AssignVariableOp_18˘AssignVariableOp_19˘AssignVariableOp_2˘AssignVariableOp_20˘AssignVariableOp_3˘AssignVariableOp_4˘AssignVariableOp_5˘AssignVariableOp_6˘AssignVariableOp_7˘AssignVariableOp_8˘AssignVariableOp_9˘	RestoreV2˘RestoreV2_1ş
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ć

valueź
Bš
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names¸
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp(assignvariableop_before_softmax_3_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOp(assignvariableop_1_before_softmax_3_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2
AssignVariableOp_2AssignVariableOp)assignvariableop_2_after_softmax_3_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3
AssignVariableOp_3AssignVariableOp'assignvariableop_3_after_softmax_3_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0	*
_output_shapes
:2

Identity_4
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13Ť
AssignVariableOp_13AssignVariableOp2assignvariableop_13_adam_before_softmax_3_kernel_mIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14Š
AssignVariableOp_14AssignVariableOp0assignvariableop_14_adam_before_softmax_3_bias_mIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15Ş
AssignVariableOp_15AssignVariableOp1assignvariableop_15_adam_after_softmax_3_kernel_mIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16¨
AssignVariableOp_16AssignVariableOp/assignvariableop_16_adam_after_softmax_3_bias_mIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17Ť
AssignVariableOp_17AssignVariableOp2assignvariableop_17_adam_before_softmax_3_kernel_vIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18Š
AssignVariableOp_18AssignVariableOp0assignvariableop_18_adam_before_softmax_3_bias_vIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19Ş
AssignVariableOp_19AssignVariableOp1assignvariableop_19_adam_after_softmax_3_kernel_vIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20¨
AssignVariableOp_20AssignVariableOp/assignvariableop_20_adam_after_softmax_3_bias_vIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20¨
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesÄ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpŹ
Identity_21Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_21š
Identity_22IdentityIdentity_21:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_22"#
identity_22Identity_22:output:0*i
_input_shapesX
V: :::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ő
ą
I__inference_after_softmax_layer_call_and_return_conditional_losses_116050

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ő
ą
I__inference_after_softmax_layer_call_and_return_conditional_losses_115830

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Č
ł
H__inference_sequential_3_layer_call_and_return_conditional_losses_115962

inputs1
-before_softmax_matmul_readvariableop_resource2
.before_softmax_biasadd_readvariableop_resource0
,after_softmax_matmul_readvariableop_resource1
-after_softmax_biasadd_readvariableop_resource
identitys
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙  2
flatten_3/Const
flatten_3/ReshapeReshapeinputsflatten_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
flatten_3/Reshapeź
$before_softmax/MatMul/ReadVariableOpReadVariableOp-before_softmax_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02&
$before_softmax/MatMul/ReadVariableOpľ
before_softmax/MatMulMatMulflatten_3/Reshape:output:0,before_softmax/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
before_softmax/MatMulş
%before_softmax/BiasAdd/ReadVariableOpReadVariableOp.before_softmax_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%before_softmax/BiasAdd/ReadVariableOpž
before_softmax/BiasAddBiasAddbefore_softmax/MatMul:product:0-before_softmax/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
before_softmax/BiasAdd
before_softmax/ReluRelubefore_softmax/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
before_softmax/Relu¸
#after_softmax/MatMul/ReadVariableOpReadVariableOp,after_softmax_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02%
#after_softmax/MatMul/ReadVariableOp¸
after_softmax/MatMulMatMul!before_softmax/Relu:activations:0+after_softmax/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
after_softmax/MatMulś
$after_softmax/BiasAdd/ReadVariableOpReadVariableOp-after_softmax_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02&
$after_softmax/BiasAdd/ReadVariableOpš
after_softmax/BiasAddBiasAddafter_softmax/MatMul:product:0,after_softmax/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
after_softmax/BiasAdd
after_softmax/SoftmaxSoftmaxafter_softmax/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
after_softmax/Softmaxs
IdentityIdentityafter_softmax/Softmax:softmax:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙:::::S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


/__inference_before_softmax_layer_call_fn_116039

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallŮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_before_softmax_layer_call_and_return_conditional_losses_1158032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Č
ł
H__inference_sequential_3_layer_call_and_return_conditional_losses_115982

inputs1
-before_softmax_matmul_readvariableop_resource2
.before_softmax_biasadd_readvariableop_resource0
,after_softmax_matmul_readvariableop_resource1
-after_softmax_biasadd_readvariableop_resource
identitys
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙  2
flatten_3/Const
flatten_3/ReshapeReshapeinputsflatten_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
flatten_3/Reshapeź
$before_softmax/MatMul/ReadVariableOpReadVariableOp-before_softmax_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02&
$before_softmax/MatMul/ReadVariableOpľ
before_softmax/MatMulMatMulflatten_3/Reshape:output:0,before_softmax/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
before_softmax/MatMulş
%before_softmax/BiasAdd/ReadVariableOpReadVariableOp.before_softmax_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%before_softmax/BiasAdd/ReadVariableOpž
before_softmax/BiasAddBiasAddbefore_softmax/MatMul:product:0-before_softmax/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
before_softmax/BiasAdd
before_softmax/ReluRelubefore_softmax/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
before_softmax/Relu¸
#after_softmax/MatMul/ReadVariableOpReadVariableOp,after_softmax_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02%
#after_softmax/MatMul/ReadVariableOp¸
after_softmax/MatMulMatMul!before_softmax/Relu:activations:0+after_softmax/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
after_softmax/MatMulś
$after_softmax/BiasAdd/ReadVariableOpReadVariableOp-after_softmax_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02&
$after_softmax/BiasAdd/ReadVariableOpš
after_softmax/BiasAddBiasAddafter_softmax/MatMul:product:0,after_softmax/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
after_softmax/BiasAdd
after_softmax/SoftmaxSoftmaxafter_softmax/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
after_softmax/Softmaxs
IdentityIdentityafter_softmax/Softmax:softmax:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙:::::S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ľ
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_115784

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0**
_input_shapes
:˙˙˙˙˙˙˙˙˙:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ô
˛
J__inference_before_softmax_layer_call_and_return_conditional_losses_116030

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

Š
-__inference_sequential_3_layer_call_fn_115919
flatten_3_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCallů
StatefulPartitionedCallStatefulPartitionedCallflatten_3_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_1159082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameflatten_3_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

 
-__inference_sequential_3_layer_call_fn_116008

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCallđ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_1159082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
î
 
$__inference_signature_wrapper_115942
flatten_3_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCallŇ
StatefulPartitionedCallStatefulPartitionedCallflatten_3_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8**
f%R#
!__inference__wrapped_model_1157742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameflatten_3_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ú
É
!__inference__wrapped_model_115774
flatten_3_input>
:sequential_3_before_softmax_matmul_readvariableop_resource?
;sequential_3_before_softmax_biasadd_readvariableop_resource=
9sequential_3_after_softmax_matmul_readvariableop_resource>
:sequential_3_after_softmax_biasadd_readvariableop_resource
identity
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙  2
sequential_3/flatten_3/Constś
sequential_3/flatten_3/ReshapeReshapeflatten_3_input%sequential_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
sequential_3/flatten_3/Reshapeă
1sequential_3/before_softmax/MatMul/ReadVariableOpReadVariableOp:sequential_3_before_softmax_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype023
1sequential_3/before_softmax/MatMul/ReadVariableOpé
"sequential_3/before_softmax/MatMulMatMul'sequential_3/flatten_3/Reshape:output:09sequential_3/before_softmax/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"sequential_3/before_softmax/MatMulá
2sequential_3/before_softmax/BiasAdd/ReadVariableOpReadVariableOp;sequential_3_before_softmax_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype024
2sequential_3/before_softmax/BiasAdd/ReadVariableOpň
#sequential_3/before_softmax/BiasAddBiasAdd,sequential_3/before_softmax/MatMul:product:0:sequential_3/before_softmax/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#sequential_3/before_softmax/BiasAdd­
 sequential_3/before_softmax/ReluRelu,sequential_3/before_softmax/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2"
 sequential_3/before_softmax/Reluß
0sequential_3/after_softmax/MatMul/ReadVariableOpReadVariableOp9sequential_3_after_softmax_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype022
0sequential_3/after_softmax/MatMul/ReadVariableOpě
!sequential_3/after_softmax/MatMulMatMul.sequential_3/before_softmax/Relu:activations:08sequential_3/after_softmax/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2#
!sequential_3/after_softmax/MatMulÝ
1sequential_3/after_softmax/BiasAdd/ReadVariableOpReadVariableOp:sequential_3_after_softmax_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype023
1sequential_3/after_softmax/BiasAdd/ReadVariableOpí
"sequential_3/after_softmax/BiasAddBiasAdd+sequential_3/after_softmax/MatMul:product:09sequential_3/after_softmax/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2$
"sequential_3/after_softmax/BiasAdd˛
"sequential_3/after_softmax/SoftmaxSoftmax+sequential_3/after_softmax/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2$
"sequential_3/after_softmax/Softmax
IdentityIdentity,sequential_3/after_softmax/Softmax:softmax:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙:::::\ X
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameflatten_3_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

Ť
H__inference_sequential_3_layer_call_and_return_conditional_losses_115862
flatten_3_input
before_softmax_115851
before_softmax_115853
after_softmax_115856
after_softmax_115858
identity˘%after_softmax/StatefulPartitionedCall˘&before_softmax/StatefulPartitionedCallż
flatten_3/PartitionedCallPartitionedCallflatten_3_input*
Tin
2*
Tout
2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1157842
flatten_3/PartitionedCall­
&before_softmax/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0before_softmax_115851before_softmax_115853*
Tin
2*
Tout
2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_before_softmax_layer_call_and_return_conditional_losses_1158032(
&before_softmax/StatefulPartitionedCall´
%after_softmax/StatefulPartitionedCallStatefulPartitionedCall/before_softmax/StatefulPartitionedCall:output:0after_softmax_115856after_softmax_115858*
Tin
2*
Tout
2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_after_softmax_layer_call_and_return_conditional_losses_1158302'
%after_softmax/StatefulPartitionedCallÓ
IdentityIdentity.after_softmax/StatefulPartitionedCall:output:0&^after_softmax/StatefulPartitionedCall'^before_softmax/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙::::2N
%after_softmax/StatefulPartitionedCall%after_softmax/StatefulPartitionedCall2P
&before_softmax/StatefulPartitionedCall&before_softmax/StatefulPartitionedCall:\ X
+
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameflatten_3_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "ŻL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ä
serving_default°
O
flatten_3_input<
!serving_default_flatten_3_input:0˙˙˙˙˙˙˙˙˙A
after_softmax0
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙
tensorflow/serving/predict:Ôy

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api
	
signatures
*F&call_and_return_all_conditional_losses
G_default_save_signature
H__call__"â
_tf_keras_sequentialĂ{"class_name": "Sequential", "name": "sequential_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_3", "layers": [{"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "before_softmax", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "after_softmax", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28]}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "before_softmax", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "after_softmax", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28]}}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ž

	variables
regularization_losses
trainable_variables
	keras_api
*I&call_and_return_all_conditional_losses
J__call__"Ż
_tf_keras_layer{"class_name": "Flatten", "name": "flatten_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "stateful": false, "config": {"name": "flatten_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
Ţ

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*K&call_and_return_all_conditional_losses
L__call__"š
_tf_keras_layer{"class_name": "Dense", "name": "before_softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "before_softmax", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
Ţ

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*M&call_and_return_all_conditional_losses
N__call__"š
_tf_keras_layer{"class_name": "Dense", "name": "after_softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "after_softmax", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}

iter

beta_1

beta_2
	decay
learning_ratem>m?m@mAvBvCvDvE"
	optimizer
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
Ę
	variables
non_trainable_variables

 layers
!metrics
"layer_metrics
regularization_losses
trainable_variables
#layer_regularization_losses
H__call__
G_default_save_signature
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
,
Oserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

	variables
$non_trainable_variables

%layers
&metrics
'layer_metrics
regularization_losses
trainable_variables
(layer_regularization_losses
J__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
+:)
2before_softmax_3/kernel
$:"2before_softmax_3/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
	variables
)non_trainable_variables

*layers
+metrics
,layer_metrics
regularization_losses
trainable_variables
-layer_regularization_losses
L__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
):'	
2after_softmax_3/kernel
": 
2after_softmax_3/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
	variables
.non_trainable_variables

/layers
0metrics
1layer_metrics
regularization_losses
trainable_variables
2layer_regularization_losses
N__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
ť
	5total
	6count
7	variables
8	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
˙
	9total
	:count
;
_fn_kwargs
<	variables
=	keras_api"¸
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
.
50
61"
trackable_list_wrapper
-
7	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
90
:1"
trackable_list_wrapper
-
<	variables"
_generic_user_object
0:.
2Adam/before_softmax_3/kernel/m
):'2Adam/before_softmax_3/bias/m
.:,	
2Adam/after_softmax_3/kernel/m
':%
2Adam/after_softmax_3/bias/m
0:.
2Adam/before_softmax_3/kernel/v
):'2Adam/before_softmax_3/bias/v
.:,	
2Adam/after_softmax_3/kernel/v
':%
2Adam/after_softmax_3/bias/v
î2ë
H__inference_sequential_3_layer_call_and_return_conditional_losses_115982
H__inference_sequential_3_layer_call_and_return_conditional_losses_115962
H__inference_sequential_3_layer_call_and_return_conditional_losses_115847
H__inference_sequential_3_layer_call_and_return_conditional_losses_115862Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ë2č
!__inference__wrapped_model_115774Â
˛
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *2˘/
-*
flatten_3_input˙˙˙˙˙˙˙˙˙
2˙
-__inference_sequential_3_layer_call_fn_115919
-__inference_sequential_3_layer_call_fn_116008
-__inference_sequential_3_layer_call_fn_115995
-__inference_sequential_3_layer_call_fn_115891Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ď2ě
E__inference_flatten_3_layer_call_and_return_conditional_losses_116014˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ô2Ń
*__inference_flatten_3_layer_call_fn_116019˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ô2ń
J__inference_before_softmax_layer_call_and_return_conditional_losses_116030˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ů2Ö
/__inference_before_softmax_layer_call_fn_116039˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ó2đ
I__inference_after_softmax_layer_call_and_return_conditional_losses_116050˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ř2Ő
.__inference_after_softmax_layer_call_fn_116059˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
;B9
$__inference_signature_wrapper_115942flatten_3_inputŠ
!__inference__wrapped_model_115774<˘9
2˘/
-*
flatten_3_input˙˙˙˙˙˙˙˙˙
Ş "=Ş:
8
after_softmax'$
after_softmax˙˙˙˙˙˙˙˙˙
Ş
I__inference_after_softmax_layer_call_and_return_conditional_losses_116050]0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "%˘"

0˙˙˙˙˙˙˙˙˙

 
.__inference_after_softmax_layer_call_fn_116059P0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙
Ź
J__inference_before_softmax_layer_call_and_return_conditional_losses_116030^0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 
/__inference_before_softmax_layer_call_fn_116039Q0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙Ś
E__inference_flatten_3_layer_call_and_return_conditional_losses_116014]3˘0
)˘&
$!
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 ~
*__inference_flatten_3_layer_call_fn_116019P3˘0
)˘&
$!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙ż
H__inference_sequential_3_layer_call_and_return_conditional_losses_115847sD˘A
:˘7
-*
flatten_3_input˙˙˙˙˙˙˙˙˙
p

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙

 ż
H__inference_sequential_3_layer_call_and_return_conditional_losses_115862sD˘A
:˘7
-*
flatten_3_input˙˙˙˙˙˙˙˙˙
p 

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙

 ś
H__inference_sequential_3_layer_call_and_return_conditional_losses_115962j;˘8
1˘.
$!
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙

 ś
H__inference_sequential_3_layer_call_and_return_conditional_losses_115982j;˘8
1˘.
$!
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙

 
-__inference_sequential_3_layer_call_fn_115891fD˘A
:˘7
-*
flatten_3_input˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙

-__inference_sequential_3_layer_call_fn_115919fD˘A
:˘7
-*
flatten_3_input˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙

-__inference_sequential_3_layer_call_fn_115995];˘8
1˘.
$!
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙

-__inference_sequential_3_layer_call_fn_116008];˘8
1˘.
$!
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙
ż
$__inference_signature_wrapper_115942O˘L
˘ 
EŞB
@
flatten_3_input-*
flatten_3_input˙˙˙˙˙˙˙˙˙"=Ş:
8
after_softmax'$
after_softmax˙˙˙˙˙˙˙˙˙
