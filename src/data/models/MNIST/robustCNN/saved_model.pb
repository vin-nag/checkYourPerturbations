ч
Њ§
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
О
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
shapeshape"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8є

conv1_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1_1/kernel
y
"conv1_1/kernel/Read/ReadVariableOpReadVariableOpconv1_1/kernel*&
_output_shapes
:*
dtype0
p
conv1_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1_1/bias
i
 conv1_1/bias/Read/ReadVariableOpReadVariableOpconv1_1/bias*
_output_shapes
:*
dtype0

conv2_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2_1/kernel
y
"conv2_1/kernel/Read/ReadVariableOpReadVariableOpconv2_1/kernel*&
_output_shapes
:*
dtype0
p
conv2_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2_1/bias
i
 conv2_1/bias/Read/ReadVariableOpReadVariableOpconv2_1/bias*
_output_shapes
:*
dtype0

before_softmax_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_namebefore_softmax_1/kernel

+before_softmax_1/kernel/Read/ReadVariableOpReadVariableOpbefore_softmax_1/kernel*
_output_shapes

:@*
dtype0

before_softmax_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_namebefore_softmax_1/bias
{
)before_softmax_1/bias/Read/ReadVariableOpReadVariableOpbefore_softmax_1/bias*
_output_shapes
:@*
dtype0

after_softmax_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*'
shared_nameafter_softmax_1/kernel

*after_softmax_1/kernel/Read/ReadVariableOpReadVariableOpafter_softmax_1/kernel*
_output_shapes

:@
*
dtype0

after_softmax_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameafter_softmax_1/bias
y
(after_softmax_1/bias/Read/ReadVariableOpReadVariableOpafter_softmax_1/bias*
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

Adam/conv1_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv1_1/kernel/m

)Adam/conv1_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1_1/kernel/m*&
_output_shapes
:*
dtype0
~
Adam/conv1_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/conv1_1/bias/m
w
'Adam/conv1_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1_1/bias/m*
_output_shapes
:*
dtype0

Adam/conv2_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2_1/kernel/m

)Adam/conv2_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2_1/kernel/m*&
_output_shapes
:*
dtype0
~
Adam/conv2_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/conv2_1/bias/m
w
'Adam/conv2_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2_1/bias/m*
_output_shapes
:*
dtype0

Adam/before_softmax_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*/
shared_name Adam/before_softmax_1/kernel/m

2Adam/before_softmax_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/before_softmax_1/kernel/m*
_output_shapes

:@*
dtype0

Adam/before_softmax_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nameAdam/before_softmax_1/bias/m

0Adam/before_softmax_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/before_softmax_1/bias/m*
_output_shapes
:@*
dtype0

Adam/after_softmax_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*.
shared_nameAdam/after_softmax_1/kernel/m

1Adam/after_softmax_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/after_softmax_1/kernel/m*
_output_shapes

:@
*
dtype0

Adam/after_softmax_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_nameAdam/after_softmax_1/bias/m

/Adam/after_softmax_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/after_softmax_1/bias/m*
_output_shapes
:
*
dtype0

Adam/conv1_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv1_1/kernel/v

)Adam/conv1_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1_1/kernel/v*&
_output_shapes
:*
dtype0
~
Adam/conv1_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/conv1_1/bias/v
w
'Adam/conv1_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1_1/bias/v*
_output_shapes
:*
dtype0

Adam/conv2_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2_1/kernel/v

)Adam/conv2_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2_1/kernel/v*&
_output_shapes
:*
dtype0
~
Adam/conv2_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/conv2_1/bias/v
w
'Adam/conv2_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2_1/bias/v*
_output_shapes
:*
dtype0

Adam/before_softmax_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*/
shared_name Adam/before_softmax_1/kernel/v

2Adam/before_softmax_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/before_softmax_1/kernel/v*
_output_shapes

:@*
dtype0

Adam/before_softmax_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nameAdam/before_softmax_1/bias/v

0Adam/before_softmax_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/before_softmax_1/bias/v*
_output_shapes
:@*
dtype0

Adam/after_softmax_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*.
shared_nameAdam/after_softmax_1/kernel/v

1Adam/after_softmax_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/after_softmax_1/kernel/v*
_output_shapes

:@
*
dtype0

Adam/after_softmax_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_nameAdam/after_softmax_1/bias/v

/Adam/after_softmax_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/after_softmax_1/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
в5
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*5
value5B5 Bљ4
Д
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
	optimizer
	regularization_losses

	variables
trainable_variables
	keras_api

signatures
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
 	variables
!	keras_api
R
"regularization_losses
#trainable_variables
$	variables
%	keras_api
h

&kernel
'bias
(regularization_losses
)trainable_variables
*	variables
+	keras_api
h

,kernel
-bias
.regularization_losses
/trainable_variables
0	variables
1	keras_api
а
2iter

3beta_1

4beta_2
	5decay
6learning_ratemjmkmlmm&mn'mo,mp-mqvrvsvtvu&vv'vw,vx-vy
 
8
0
1
2
3
&4
'5
,6
-7
8
0
1
2
3
&4
'5
,6
-7
­

7layers
	regularization_losses
8layer_regularization_losses
9non_trainable_variables
:layer_metrics
;metrics

	variables
trainable_variables
 
ZX
VARIABLE_VALUEconv1_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv1_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­

<layers
regularization_losses
=layer_regularization_losses
trainable_variables
>layer_metrics
?metrics
	variables
@non_trainable_variables
 
 
 
­

Alayers
regularization_losses
Blayer_regularization_losses
trainable_variables
Clayer_metrics
Dmetrics
	variables
Enon_trainable_variables
ZX
VARIABLE_VALUEconv2_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­

Flayers
regularization_losses
Glayer_regularization_losses
trainable_variables
Hlayer_metrics
Imetrics
	variables
Jnon_trainable_variables
 
 
 
­

Klayers
regularization_losses
Llayer_regularization_losses
trainable_variables
Mlayer_metrics
Nmetrics
 	variables
Onon_trainable_variables
 
 
 
­

Players
"regularization_losses
Qlayer_regularization_losses
#trainable_variables
Rlayer_metrics
Smetrics
$	variables
Tnon_trainable_variables
ca
VARIABLE_VALUEbefore_softmax_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEbefore_softmax_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

&0
'1

&0
'1
­

Ulayers
(regularization_losses
Vlayer_regularization_losses
)trainable_variables
Wlayer_metrics
Xmetrics
*	variables
Ynon_trainable_variables
b`
VARIABLE_VALUEafter_softmax_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEafter_softmax_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

,0
-1

,0
-1
­

Zlayers
.regularization_losses
[layer_regularization_losses
/trainable_variables
\layer_metrics
]metrics
0	variables
^non_trainable_variables
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
1
0
1
2
3
4
5
6
 
 
 

_0
`1
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
 
4
	atotal
	bcount
c	variables
d	keras_api
D
	etotal
	fcount
g
_fn_kwargs
h	variables
i	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

a0
b1

c	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

e0
f1

h	variables
}{
VARIABLE_VALUEAdam/conv1_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv1_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv2_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/before_softmax_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/before_softmax_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/after_softmax_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/after_softmax_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv1_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv1_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv2_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/before_softmax_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/before_softmax_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/after_softmax_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/after_softmax_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_inputPlaceholder*/
_output_shapes
:џџџџџџџџџ*
dtype0*$
shape:џџџџџџџџџ
К
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputconv1_1/kernelconv1_1/biasconv2_1/kernelconv2_1/biasbefore_softmax_1/kernelbefore_softmax_1/biasafter_softmax_1/kernelafter_softmax_1/bias*
Tin
2	*
Tout
2*'
_output_shapes
:џџџџџџџџџ
**
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*,
f'R%
#__inference_signature_wrapper_98998
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
г
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"conv1_1/kernel/Read/ReadVariableOp conv1_1/bias/Read/ReadVariableOp"conv2_1/kernel/Read/ReadVariableOp conv2_1/bias/Read/ReadVariableOp+before_softmax_1/kernel/Read/ReadVariableOp)before_softmax_1/bias/Read/ReadVariableOp*after_softmax_1/kernel/Read/ReadVariableOp(after_softmax_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)Adam/conv1_1/kernel/m/Read/ReadVariableOp'Adam/conv1_1/bias/m/Read/ReadVariableOp)Adam/conv2_1/kernel/m/Read/ReadVariableOp'Adam/conv2_1/bias/m/Read/ReadVariableOp2Adam/before_softmax_1/kernel/m/Read/ReadVariableOp0Adam/before_softmax_1/bias/m/Read/ReadVariableOp1Adam/after_softmax_1/kernel/m/Read/ReadVariableOp/Adam/after_softmax_1/bias/m/Read/ReadVariableOp)Adam/conv1_1/kernel/v/Read/ReadVariableOp'Adam/conv1_1/bias/v/Read/ReadVariableOp)Adam/conv2_1/kernel/v/Read/ReadVariableOp'Adam/conv2_1/bias/v/Read/ReadVariableOp2Adam/before_softmax_1/kernel/v/Read/ReadVariableOp0Adam/before_softmax_1/bias/v/Read/ReadVariableOp1Adam/after_softmax_1/kernel/v/Read/ReadVariableOp/Adam/after_softmax_1/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*'
f"R 
__inference__traced_save_99287
К
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1_1/kernelconv1_1/biasconv2_1/kernelconv2_1/biasbefore_softmax_1/kernelbefore_softmax_1/biasafter_softmax_1/kernelafter_softmax_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv1_1/kernel/mAdam/conv1_1/bias/mAdam/conv2_1/kernel/mAdam/conv2_1/bias/mAdam/before_softmax_1/kernel/mAdam/before_softmax_1/bias/mAdam/after_softmax_1/kernel/mAdam/after_softmax_1/bias/mAdam/conv1_1/kernel/vAdam/conv1_1/bias/vAdam/conv2_1/kernel/vAdam/conv2_1/bias/vAdam/before_softmax_1/kernel/vAdam/before_softmax_1/bias/vAdam/after_softmax_1/kernel/vAdam/after_softmax_1/bias/v*-
Tin&
$2"*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8**
f%R#
!__inference__traced_restore_99398ё


к
,__inference_sequential_1_layer_call_fn_98919	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*'
_output_shapes
:џџџџџџџџџ
**
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_989002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:џџџџџџџџџ

_user_specified_nameinput:
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
: 
б
 
G__inference_sequential_1_layer_call_and_return_conditional_losses_98870	
input
conv1_98846
conv1_98848
conv2_98852
conv2_98854
before_softmax_98859
before_softmax_98861
after_softmax_98864
after_softmax_98866
identityЂ%after_softmax/StatefulPartitionedCallЂ&before_softmax/StatefulPartitionedCallЂconv1/StatefulPartitionedCallЂconv2/StatefulPartitionedCallъ
conv1/StatefulPartitionedCallStatefulPartitionedCallinputconv1_98846conv1_98848*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_987032
conv1/StatefulPartitionedCallЭ
mp1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*G
fBR@
>__inference_mp1_layer_call_and_return_conditional_losses_987192
mp1/PartitionedCall
conv2/StatefulPartitionedCallStatefulPartitionedCallmp1/PartitionedCall:output:0conv2_98852conv2_98854*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*I
fDRB
@__inference_conv2_layer_call_and_return_conditional_losses_987362
conv2/StatefulPartitionedCallЭ
mp2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*G
fBR@
>__inference_mp2_layer_call_and_return_conditional_losses_987522
mp2/PartitionedCallЧ
flatten/PartitionedCallPartitionedCallmp2/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_987802
flatten/PartitionedCallЊ
&before_softmax/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0before_softmax_98859before_softmax_98861*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_before_softmax_layer_call_and_return_conditional_losses_987992(
&before_softmax/StatefulPartitionedCallД
%after_softmax/StatefulPartitionedCallStatefulPartitionedCall/before_softmax/StatefulPartitionedCall:output:0after_softmax_98864after_softmax_98866*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*Q
fLRJ
H__inference_after_softmax_layer_call_and_return_conditional_losses_988262'
%after_softmax/StatefulPartitionedCall
IdentityIdentity.after_softmax/StatefulPartitionedCall:output:0&^after_softmax/StatefulPartitionedCall'^before_softmax/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ::::::::2N
%after_softmax/StatefulPartitionedCall%after_softmax/StatefulPartitionedCall2P
&before_softmax/StatefulPartitionedCall&before_softmax/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall:V R
/
_output_shapes
:џџџџџџџџџ

_user_specified_nameinput:
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
: 
Ў

Ј
@__inference_conv1_layer_call_and_return_conditional_losses_98703

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
џ
C
'__inference_flatten_layer_call_fn_99121

inputs
identityЁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_987802
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


к
,__inference_sequential_1_layer_call_fn_98967	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*'
_output_shapes
:џџџџџџџџџ
**
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_989482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:џџџџџџџџџ

_user_specified_nameinput:
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
: 
И
^
B__inference_flatten_layer_call_and_return_conditional_losses_98780

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


л
,__inference_sequential_1_layer_call_fn_99089

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*'
_output_shapes
:џџџџџџџџџ
**
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_989002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
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
: 
д
Ё
G__inference_sequential_1_layer_call_and_return_conditional_losses_98948

inputs
conv1_98924
conv1_98926
conv2_98930
conv2_98932
before_softmax_98937
before_softmax_98939
after_softmax_98942
after_softmax_98944
identityЂ%after_softmax/StatefulPartitionedCallЂ&before_softmax/StatefulPartitionedCallЂconv1/StatefulPartitionedCallЂconv2/StatefulPartitionedCallы
conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_98924conv1_98926*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_987032
conv1/StatefulPartitionedCallЭ
mp1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*G
fBR@
>__inference_mp1_layer_call_and_return_conditional_losses_987192
mp1/PartitionedCall
conv2/StatefulPartitionedCallStatefulPartitionedCallmp1/PartitionedCall:output:0conv2_98930conv2_98932*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*I
fDRB
@__inference_conv2_layer_call_and_return_conditional_losses_987362
conv2/StatefulPartitionedCallЭ
mp2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*G
fBR@
>__inference_mp2_layer_call_and_return_conditional_losses_987522
mp2/PartitionedCallЧ
flatten/PartitionedCallPartitionedCallmp2/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_987802
flatten/PartitionedCallЊ
&before_softmax/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0before_softmax_98937before_softmax_98939*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_before_softmax_layer_call_and_return_conditional_losses_987992(
&before_softmax/StatefulPartitionedCallД
%after_softmax/StatefulPartitionedCallStatefulPartitionedCall/before_softmax/StatefulPartitionedCall:output:0after_softmax_98942after_softmax_98944*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*Q
fLRJ
H__inference_after_softmax_layer_call_and_return_conditional_losses_988262'
%after_softmax/StatefulPartitionedCall
IdentityIdentity.after_softmax/StatefulPartitionedCall:output:0&^after_softmax/StatefulPartitionedCall'^before_softmax/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ::::::::2N
%after_softmax/StatefulPartitionedCall%after_softmax/StatefulPartitionedCall2P
&before_softmax/StatefulPartitionedCall&before_softmax/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
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
: 


л
,__inference_sequential_1_layer_call_fn_99110

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*'
_output_shapes
:џџџџџџџџџ
**
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_989482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
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
: 
є
Z
>__inference_mp1_layer_call_and_return_conditional_losses_98719

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
І&
м
G__inference_sequential_1_layer_call_and_return_conditional_losses_99068

inputs(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource1
-before_softmax_matmul_readvariableop_resource2
.before_softmax_biasadd_readvariableop_resource0
,after_softmax_matmul_readvariableop_resource1
-after_softmax_biasadd_readvariableop_resource
identityЇ
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOpЕ
conv1/Conv2DConv2Dinputs#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv1/Conv2D
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1/BiasAdd/ReadVariableOp 
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv1/BiasAddr

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

conv1/ReluЌ
mp1/MaxPoolMaxPoolconv1/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2
mp1/MaxPoolЇ
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2/Conv2D/ReadVariableOpФ
conv2/Conv2DConv2Dmp1/MaxPool:output:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
conv2/Conv2D
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2/BiasAdd/ReadVariableOp 
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv2/BiasAddЊ
mp2/MaxPoolMaxPoolconv2/BiasAdd:output:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2
mp2/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
flatten/Const
flatten/ReshapeReshapemp2/MaxPool:output:0flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
flatten/ReshapeК
$before_softmax/MatMul/ReadVariableOpReadVariableOp-before_softmax_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02&
$before_softmax/MatMul/ReadVariableOpВ
before_softmax/MatMulMatMulflatten/Reshape:output:0,before_softmax/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
before_softmax/MatMulЙ
%before_softmax/BiasAdd/ReadVariableOpReadVariableOp.before_softmax_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%before_softmax/BiasAdd/ReadVariableOpН
before_softmax/BiasAddBiasAddbefore_softmax/MatMul:product:0-before_softmax/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
before_softmax/BiasAdd
before_softmax/ReluRelubefore_softmax/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
before_softmax/ReluЗ
#after_softmax/MatMul/ReadVariableOpReadVariableOp,after_softmax_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02%
#after_softmax/MatMul/ReadVariableOpИ
after_softmax/MatMulMatMul!before_softmax/Relu:activations:0+after_softmax/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
after_softmax/MatMulЖ
$after_softmax/BiasAdd/ReadVariableOpReadVariableOp-after_softmax_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02&
$after_softmax/BiasAdd/ReadVariableOpЙ
after_softmax/BiasAddBiasAddafter_softmax/MatMul:product:0,after_softmax/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
after_softmax/BiasAdd
after_softmax/SoftmaxSoftmaxafter_softmax/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
after_softmax/Softmaxs
IdentityIdentityafter_softmax/Softmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ:::::::::W S
/
_output_shapes
:џџџџџџџџџ
 
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
: 
б	
б
#__inference_signature_wrapper_98998	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*'
_output_shapes
:џџџџџџџџџ
**
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*)
f$R"
 __inference__wrapped_model_986912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:џџџџџџџџџ

_user_specified_nameinput:
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
: 
д
Ё
G__inference_sequential_1_layer_call_and_return_conditional_losses_98900

inputs
conv1_98876
conv1_98878
conv2_98882
conv2_98884
before_softmax_98889
before_softmax_98891
after_softmax_98894
after_softmax_98896
identityЂ%after_softmax/StatefulPartitionedCallЂ&before_softmax/StatefulPartitionedCallЂconv1/StatefulPartitionedCallЂconv2/StatefulPartitionedCallы
conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_98876conv1_98878*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_987032
conv1/StatefulPartitionedCallЭ
mp1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*G
fBR@
>__inference_mp1_layer_call_and_return_conditional_losses_987192
mp1/PartitionedCall
conv2/StatefulPartitionedCallStatefulPartitionedCallmp1/PartitionedCall:output:0conv2_98882conv2_98884*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*I
fDRB
@__inference_conv2_layer_call_and_return_conditional_losses_987362
conv2/StatefulPartitionedCallЭ
mp2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*G
fBR@
>__inference_mp2_layer_call_and_return_conditional_losses_987522
mp2/PartitionedCallЧ
flatten/PartitionedCallPartitionedCallmp2/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_987802
flatten/PartitionedCallЊ
&before_softmax/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0before_softmax_98889before_softmax_98891*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_before_softmax_layer_call_and_return_conditional_losses_987992(
&before_softmax/StatefulPartitionedCallД
%after_softmax/StatefulPartitionedCallStatefulPartitionedCall/before_softmax/StatefulPartitionedCall:output:0after_softmax_98894after_softmax_98896*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*Q
fLRJ
H__inference_after_softmax_layer_call_and_return_conditional_losses_988262'
%after_softmax/StatefulPartitionedCall
IdentityIdentity.after_softmax/StatefulPartitionedCall:output:0&^after_softmax/StatefulPartitionedCall'^before_softmax/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ::::::::2N
%after_softmax/StatefulPartitionedCall%after_softmax/StatefulPartitionedCall2P
&before_softmax/StatefulPartitionedCall&before_softmax/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
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
: 
й
z
%__inference_conv2_layer_call_fn_98746

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*I
fDRB
@__inference_conv2_layer_call_and_return_conditional_losses_987362
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
И	
Ј
@__inference_conv2_layer_call_and_return_conditional_losses_98736

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЖ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
є
Z
>__inference_mp2_layer_call_and_return_conditional_losses_98752

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
И
^
B__inference_flatten_layer_call_and_return_conditional_losses_99116

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


.__inference_before_softmax_layer_call_fn_99141

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_before_softmax_layer_call_and_return_conditional_losses_987992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ё
А
H__inference_after_softmax_layer_call_and_return_conditional_losses_99152

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
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
:џџџџџџџџџ
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:::O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
а.

 __inference__wrapped_model_98691	
input5
1sequential_1_conv1_conv2d_readvariableop_resource6
2sequential_1_conv1_biasadd_readvariableop_resource5
1sequential_1_conv2_conv2d_readvariableop_resource6
2sequential_1_conv2_biasadd_readvariableop_resource>
:sequential_1_before_softmax_matmul_readvariableop_resource?
;sequential_1_before_softmax_biasadd_readvariableop_resource=
9sequential_1_after_softmax_matmul_readvariableop_resource>
:sequential_1_after_softmax_biasadd_readvariableop_resource
identityЮ
(sequential_1/conv1/Conv2D/ReadVariableOpReadVariableOp1sequential_1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(sequential_1/conv1/Conv2D/ReadVariableOpл
sequential_1/conv1/Conv2DConv2Dinput0sequential_1/conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
sequential_1/conv1/Conv2DХ
)sequential_1/conv1/BiasAdd/ReadVariableOpReadVariableOp2sequential_1_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential_1/conv1/BiasAdd/ReadVariableOpд
sequential_1/conv1/BiasAddBiasAdd"sequential_1/conv1/Conv2D:output:01sequential_1/conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
sequential_1/conv1/BiasAdd
sequential_1/conv1/ReluRelu#sequential_1/conv1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
sequential_1/conv1/Reluг
sequential_1/mp1/MaxPoolMaxPool%sequential_1/conv1/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2
sequential_1/mp1/MaxPoolЮ
(sequential_1/conv2/Conv2D/ReadVariableOpReadVariableOp1sequential_1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(sequential_1/conv2/Conv2D/ReadVariableOpј
sequential_1/conv2/Conv2DConv2D!sequential_1/mp1/MaxPool:output:00sequential_1/conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
sequential_1/conv2/Conv2DХ
)sequential_1/conv2/BiasAdd/ReadVariableOpReadVariableOp2sequential_1_conv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential_1/conv2/BiasAdd/ReadVariableOpд
sequential_1/conv2/BiasAddBiasAdd"sequential_1/conv2/Conv2D:output:01sequential_1/conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
sequential_1/conv2/BiasAddб
sequential_1/mp2/MaxPoolMaxPool#sequential_1/conv2/BiasAdd:output:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2
sequential_1/mp2/MaxPool
sequential_1/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
sequential_1/flatten/ConstС
sequential_1/flatten/ReshapeReshape!sequential_1/mp2/MaxPool:output:0#sequential_1/flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_1/flatten/Reshapeс
1sequential_1/before_softmax/MatMul/ReadVariableOpReadVariableOp:sequential_1_before_softmax_matmul_readvariableop_resource*
_output_shapes

:@*
dtype023
1sequential_1/before_softmax/MatMul/ReadVariableOpц
"sequential_1/before_softmax/MatMulMatMul%sequential_1/flatten/Reshape:output:09sequential_1/before_softmax/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2$
"sequential_1/before_softmax/MatMulр
2sequential_1/before_softmax/BiasAdd/ReadVariableOpReadVariableOp;sequential_1_before_softmax_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype024
2sequential_1/before_softmax/BiasAdd/ReadVariableOpё
#sequential_1/before_softmax/BiasAddBiasAdd,sequential_1/before_softmax/MatMul:product:0:sequential_1/before_softmax/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2%
#sequential_1/before_softmax/BiasAddЌ
 sequential_1/before_softmax/ReluRelu,sequential_1/before_softmax/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2"
 sequential_1/before_softmax/Reluо
0sequential_1/after_softmax/MatMul/ReadVariableOpReadVariableOp9sequential_1_after_softmax_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype022
0sequential_1/after_softmax/MatMul/ReadVariableOpь
!sequential_1/after_softmax/MatMulMatMul.sequential_1/before_softmax/Relu:activations:08sequential_1/after_softmax/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2#
!sequential_1/after_softmax/MatMulн
1sequential_1/after_softmax/BiasAdd/ReadVariableOpReadVariableOp:sequential_1_after_softmax_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype023
1sequential_1/after_softmax/BiasAdd/ReadVariableOpэ
"sequential_1/after_softmax/BiasAddBiasAdd+sequential_1/after_softmax/MatMul:product:09sequential_1/after_softmax/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2$
"sequential_1/after_softmax/BiasAddВ
"sequential_1/after_softmax/SoftmaxSoftmax+sequential_1/after_softmax/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
2$
"sequential_1/after_softmax/Softmax
IdentityIdentity,sequential_1/after_softmax/Softmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ:::::::::V R
/
_output_shapes
:џџџџџџџџџ

_user_specified_nameinput:
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
: 
Х
л
!__inference__traced_restore_99398
file_prefix#
assignvariableop_conv1_1_kernel#
assignvariableop_1_conv1_1_bias%
!assignvariableop_2_conv2_1_kernel#
assignvariableop_3_conv2_1_bias.
*assignvariableop_4_before_softmax_1_kernel,
(assignvariableop_5_before_softmax_1_bias-
)assignvariableop_6_after_softmax_1_kernel+
'assignvariableop_7_after_softmax_1_bias 
assignvariableop_8_adam_iter"
assignvariableop_9_adam_beta_1#
assignvariableop_10_adam_beta_2"
assignvariableop_11_adam_decay*
&assignvariableop_12_adam_learning_rate
assignvariableop_13_total
assignvariableop_14_count
assignvariableop_15_total_1
assignvariableop_16_count_1-
)assignvariableop_17_adam_conv1_1_kernel_m+
'assignvariableop_18_adam_conv1_1_bias_m-
)assignvariableop_19_adam_conv2_1_kernel_m+
'assignvariableop_20_adam_conv2_1_bias_m6
2assignvariableop_21_adam_before_softmax_1_kernel_m4
0assignvariableop_22_adam_before_softmax_1_bias_m5
1assignvariableop_23_adam_after_softmax_1_kernel_m3
/assignvariableop_24_adam_after_softmax_1_bias_m-
)assignvariableop_25_adam_conv1_1_kernel_v+
'assignvariableop_26_adam_conv1_1_bias_v-
)assignvariableop_27_adam_conv2_1_kernel_v+
'assignvariableop_28_adam_conv2_1_bias_v6
2assignvariableop_29_adam_before_softmax_1_kernel_v4
0assignvariableop_30_adam_before_softmax_1_bias_v5
1assignvariableop_31_adam_after_softmax_1_kernel_v3
/assignvariableop_32_adam_after_softmax_1_bias_v
identity_34ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Ђ	RestoreV2ЂRestoreV2_1Ў
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*К
valueАB­!B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesа
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesг
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
:::::::::::::::::::::::::::::::::*/
dtypes%
#2!	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_conv1_1_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1_1_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2
AssignVariableOp_2AssignVariableOp!assignvariableop_2_conv2_1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv2_1_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4 
AssignVariableOp_4AssignVariableOp*assignvariableop_4_before_softmax_1_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5
AssignVariableOp_5AssignVariableOp(assignvariableop_5_before_softmax_1_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOp)assignvariableop_6_after_softmax_1_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7
AssignVariableOp_7AssignVariableOp'assignvariableop_7_after_softmax_1_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0	*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17Ђ
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_conv1_1_kernel_mIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18 
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_conv1_1_bias_mIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19Ђ
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_conv2_1_kernel_mIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20 
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_conv2_1_bias_mIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21Ћ
AssignVariableOp_21AssignVariableOp2assignvariableop_21_adam_before_softmax_1_kernel_mIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22Љ
AssignVariableOp_22AssignVariableOp0assignvariableop_22_adam_before_softmax_1_bias_mIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23Њ
AssignVariableOp_23AssignVariableOp1assignvariableop_23_adam_after_softmax_1_kernel_mIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24Ј
AssignVariableOp_24AssignVariableOp/assignvariableop_24_adam_after_softmax_1_bias_mIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25Ђ
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_conv1_1_kernel_vIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26 
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_conv1_1_bias_vIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27Ђ
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_conv2_1_kernel_vIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28 
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_conv2_1_bias_vIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29Ћ
AssignVariableOp_29AssignVariableOp2assignvariableop_29_adam_before_softmax_1_kernel_vIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30Љ
AssignVariableOp_30AssignVariableOp0assignvariableop_30_adam_before_softmax_1_bias_vIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31Њ
AssignVariableOp_31AssignVariableOp1assignvariableop_31_adam_after_softmax_1_kernel_vIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32Ј
AssignVariableOp_32AssignVariableOp/assignvariableop_32_adam_after_softmax_1_bias_vIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32Ј
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
RestoreV2_1/shape_and_slicesФ
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
NoOpД
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_33С
Identity_34IdentityIdentity_33:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_34"#
identity_34Identity_34:output:0*
_input_shapes
: :::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: 


-__inference_after_softmax_layer_call_fn_99161

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*Q
fLRJ
H__inference_after_softmax_layer_call_and_return_conditional_losses_988262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ё
А
H__inference_after_softmax_layer_call_and_return_conditional_losses_98826

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
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
:џџџџџџџџџ
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:::O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ъ
Б
I__inference_before_softmax_layer_call_and_return_conditional_losses_98799

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:::O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
й
z
%__inference_conv1_layer_call_fn_98713

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_987032
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
б
 
G__inference_sequential_1_layer_call_and_return_conditional_losses_98843	
input
conv1_98762
conv1_98764
conv2_98768
conv2_98770
before_softmax_98810
before_softmax_98812
after_softmax_98837
after_softmax_98839
identityЂ%after_softmax/StatefulPartitionedCallЂ&before_softmax/StatefulPartitionedCallЂconv1/StatefulPartitionedCallЂconv2/StatefulPartitionedCallъ
conv1/StatefulPartitionedCallStatefulPartitionedCallinputconv1_98762conv1_98764*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_987032
conv1/StatefulPartitionedCallЭ
mp1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*G
fBR@
>__inference_mp1_layer_call_and_return_conditional_losses_987192
mp1/PartitionedCall
conv2/StatefulPartitionedCallStatefulPartitionedCallmp1/PartitionedCall:output:0conv2_98768conv2_98770*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*I
fDRB
@__inference_conv2_layer_call_and_return_conditional_losses_987362
conv2/StatefulPartitionedCallЭ
mp2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*G
fBR@
>__inference_mp2_layer_call_and_return_conditional_losses_987522
mp2/PartitionedCallЧ
flatten/PartitionedCallPartitionedCallmp2/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_987802
flatten/PartitionedCallЊ
&before_softmax/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0before_softmax_98810before_softmax_98812*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_before_softmax_layer_call_and_return_conditional_losses_987992(
&before_softmax/StatefulPartitionedCallД
%after_softmax/StatefulPartitionedCallStatefulPartitionedCall/before_softmax/StatefulPartitionedCall:output:0after_softmax_98837after_softmax_98839*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*Q
fLRJ
H__inference_after_softmax_layer_call_and_return_conditional_losses_988262'
%after_softmax/StatefulPartitionedCall
IdentityIdentity.after_softmax/StatefulPartitionedCall:output:0&^after_softmax/StatefulPartitionedCall'^before_softmax/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ::::::::2N
%after_softmax/StatefulPartitionedCall%after_softmax/StatefulPartitionedCall2P
&before_softmax/StatefulPartitionedCall&before_softmax/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall:V R
/
_output_shapes
:џџџџџџџџџ

_user_specified_nameinput:
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
: 
є
?
#__inference_mp1_layer_call_fn_98725

inputs
identityР
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*G
fBR@
>__inference_mp1_layer_call_and_return_conditional_losses_987192
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
њP
Ѓ
__inference__traced_save_99287
file_prefix-
)savev2_conv1_1_kernel_read_readvariableop+
'savev2_conv1_1_bias_read_readvariableop-
)savev2_conv2_1_kernel_read_readvariableop+
'savev2_conv2_1_bias_read_readvariableop6
2savev2_before_softmax_1_kernel_read_readvariableop4
0savev2_before_softmax_1_bias_read_readvariableop5
1savev2_after_softmax_1_kernel_read_readvariableop3
/savev2_after_softmax_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_adam_conv1_1_kernel_m_read_readvariableop2
.savev2_adam_conv1_1_bias_m_read_readvariableop4
0savev2_adam_conv2_1_kernel_m_read_readvariableop2
.savev2_adam_conv2_1_bias_m_read_readvariableop=
9savev2_adam_before_softmax_1_kernel_m_read_readvariableop;
7savev2_adam_before_softmax_1_bias_m_read_readvariableop<
8savev2_adam_after_softmax_1_kernel_m_read_readvariableop:
6savev2_adam_after_softmax_1_bias_m_read_readvariableop4
0savev2_adam_conv1_1_kernel_v_read_readvariableop2
.savev2_adam_conv1_1_bias_v_read_readvariableop4
0savev2_adam_conv2_1_kernel_v_read_readvariableop2
.savev2_adam_conv2_1_bias_v_read_readvariableop=
9savev2_adam_before_softmax_1_kernel_v_read_readvariableop;
7savev2_adam_before_softmax_1_bias_v_read_readvariableop<
8savev2_adam_after_softmax_1_kernel_v_read_readvariableop:
6savev2_adam_after_softmax_1_bias_v_read_readvariableop
savev2_1_const

identity_1ЂMergeV2CheckpointsЂSaveV2ЂSaveV2_1
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
value3B1 B+_temp_de702714ab904c0a839c20b28d081165/part2	
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
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameЈ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*К
valueАB­!B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesЪ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesъ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_conv1_1_kernel_read_readvariableop'savev2_conv1_1_bias_read_readvariableop)savev2_conv2_1_kernel_read_readvariableop'savev2_conv2_1_bias_read_readvariableop2savev2_before_softmax_1_kernel_read_readvariableop0savev2_before_softmax_1_bias_read_readvariableop1savev2_after_softmax_1_kernel_read_readvariableop/savev2_after_softmax_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adam_conv1_1_kernel_m_read_readvariableop.savev2_adam_conv1_1_bias_m_read_readvariableop0savev2_adam_conv2_1_kernel_m_read_readvariableop.savev2_adam_conv2_1_bias_m_read_readvariableop9savev2_adam_before_softmax_1_kernel_m_read_readvariableop7savev2_adam_before_softmax_1_bias_m_read_readvariableop8savev2_adam_after_softmax_1_kernel_m_read_readvariableop6savev2_adam_after_softmax_1_bias_m_read_readvariableop0savev2_adam_conv1_1_kernel_v_read_readvariableop.savev2_adam_conv1_1_bias_v_read_readvariableop0savev2_adam_conv2_1_kernel_v_read_readvariableop.savev2_adam_conv2_1_bias_v_read_readvariableop9savev2_adam_before_softmax_1_kernel_v_read_readvariableop7savev2_adam_before_softmax_1_bias_v_read_readvariableop8savev2_adam_after_softmax_1_kernel_v_read_readvariableop6savev2_adam_after_softmax_1_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 */
dtypes%
#2!	2
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardЌ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1Ђ
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
SaveV2_1/shape_and_slicesЯ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1у
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЌ
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

identity_1Identity_1:output:0*
_input_shapes
: :::::@:@:@
:
: : : : : : : : : :::::@:@:@
:
:::::@:@:@
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@
: 

_output_shapes
:
:	
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
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@
: 

_output_shapes
:
:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
:@:$  

_output_shapes

:@
: !

_output_shapes
:
:"

_output_shapes
: 
І&
м
G__inference_sequential_1_layer_call_and_return_conditional_losses_99033

inputs(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource1
-before_softmax_matmul_readvariableop_resource2
.before_softmax_biasadd_readvariableop_resource0
,after_softmax_matmul_readvariableop_resource1
-after_softmax_biasadd_readvariableop_resource
identityЇ
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOpЕ
conv1/Conv2DConv2Dinputs#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv1/Conv2D
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1/BiasAdd/ReadVariableOp 
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv1/BiasAddr

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

conv1/ReluЌ
mp1/MaxPoolMaxPoolconv1/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2
mp1/MaxPoolЇ
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2/Conv2D/ReadVariableOpФ
conv2/Conv2DConv2Dmp1/MaxPool:output:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
conv2/Conv2D
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2/BiasAdd/ReadVariableOp 
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv2/BiasAddЊ
mp2/MaxPoolMaxPoolconv2/BiasAdd:output:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2
mp2/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
flatten/Const
flatten/ReshapeReshapemp2/MaxPool:output:0flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
flatten/ReshapeК
$before_softmax/MatMul/ReadVariableOpReadVariableOp-before_softmax_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02&
$before_softmax/MatMul/ReadVariableOpВ
before_softmax/MatMulMatMulflatten/Reshape:output:0,before_softmax/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
before_softmax/MatMulЙ
%before_softmax/BiasAdd/ReadVariableOpReadVariableOp.before_softmax_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%before_softmax/BiasAdd/ReadVariableOpН
before_softmax/BiasAddBiasAddbefore_softmax/MatMul:product:0-before_softmax/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
before_softmax/BiasAdd
before_softmax/ReluRelubefore_softmax/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
before_softmax/ReluЗ
#after_softmax/MatMul/ReadVariableOpReadVariableOp,after_softmax_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02%
#after_softmax/MatMul/ReadVariableOpИ
after_softmax/MatMulMatMul!before_softmax/Relu:activations:0+after_softmax/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
after_softmax/MatMulЖ
$after_softmax/BiasAdd/ReadVariableOpReadVariableOp-after_softmax_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02&
$after_softmax/BiasAdd/ReadVariableOpЙ
after_softmax/BiasAddBiasAddafter_softmax/MatMul:product:0,after_softmax/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
after_softmax/BiasAdd
after_softmax/SoftmaxSoftmaxafter_softmax/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
2
after_softmax/Softmaxs
IdentityIdentityafter_softmax/Softmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ:::::::::W S
/
_output_shapes
:џџџџџџџџџ
 
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
: 
є
?
#__inference_mp2_layer_call_fn_98758

inputs
identityР
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*G
fBR@
>__inference_mp2_layer_call_and_return_conditional_losses_987522
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ъ
Б
I__inference_before_softmax_layer_call_and_return_conditional_losses_99132

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:::O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: "ЏL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Д
serving_default 
?
input6
serving_default_input:0џџџџџџџџџA
after_softmax0
StatefulPartitionedCall:0џџџџџџџџџ
tensorflow/serving/predict:ищ
ђ9
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
	optimizer
	regularization_losses

	variables
trainable_variables
	keras_api

signatures
z__call__
{_default_save_signature
*|&call_and_return_all_conditional_losses"ф6
_tf_keras_sequentialХ6{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_1", "layers": [{"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}}}, {"class_name": "MaxPooling2D", "config": {"name": "mp1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "mp2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "before_softmax", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "after_softmax", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}}}, {"class_name": "MaxPooling2D", "config": {"name": "mp1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "mp2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "before_softmax", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "after_softmax", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
К	

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
}__call__
*~&call_and_return_all_conditional_losses"
_tf_keras_layerћ{"class_name": "Conv2D", "name": "conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}
Х
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"Е
_tf_keras_layer{"class_name": "MaxPooling2D", "name": "mp1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "mp1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
П	

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layerў{"class_name": "Conv2D", "name": "conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 20]}}
Ц
regularization_losses
trainable_variables
 	variables
!	keras_api
__call__
+&call_and_return_all_conditional_losses"Е
_tf_keras_layer{"class_name": "MaxPooling2D", "name": "mp2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "mp2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
С
"regularization_losses
#trainable_variables
$	variables
%	keras_api
__call__
+&call_and_return_all_conditional_losses"А
_tf_keras_layer{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
н

&kernel
'bias
(regularization_losses
)trainable_variables
*	variables
+	keras_api
__call__
+&call_and_return_all_conditional_losses"Ж
_tf_keras_layer{"class_name": "Dense", "name": "before_softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "before_softmax", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
о

,kernel
-bias
.regularization_losses
/trainable_variables
0	variables
1	keras_api
__call__
+&call_and_return_all_conditional_losses"З
_tf_keras_layer{"class_name": "Dense", "name": "after_softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "after_softmax", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
у
2iter

3beta_1

4beta_2
	5decay
6learning_ratemjmkmlmm&mn'mo,mp-mqvrvsvtvu&vv'vw,vx-vy"
	optimizer
 "
trackable_list_wrapper
X
0
1
2
3
&4
'5
,6
-7"
trackable_list_wrapper
X
0
1
2
3
&4
'5
,6
-7"
trackable_list_wrapper
Ъ

7layers
	regularization_losses
8layer_regularization_losses
9non_trainable_variables
:layer_metrics
;metrics

	variables
trainable_variables
z__call__
{_default_save_signature
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
(:&2conv1_1/kernel
:2conv1_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

<layers
regularization_losses
=layer_regularization_losses
trainable_variables
>layer_metrics
?metrics
	variables
@non_trainable_variables
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Џ

Alayers
regularization_losses
Blayer_regularization_losses
trainable_variables
Clayer_metrics
Dmetrics
	variables
Enon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
(:&2conv2_1/kernel
:2conv2_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
А

Flayers
regularization_losses
Glayer_regularization_losses
trainable_variables
Hlayer_metrics
Imetrics
	variables
Jnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А

Klayers
regularization_losses
Llayer_regularization_losses
trainable_variables
Mlayer_metrics
Nmetrics
 	variables
Onon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А

Players
"regularization_losses
Qlayer_regularization_losses
#trainable_variables
Rlayer_metrics
Smetrics
$	variables
Tnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
):'@2before_softmax_1/kernel
#:!@2before_softmax_1/bias
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
А

Ulayers
(regularization_losses
Vlayer_regularization_losses
)trainable_variables
Wlayer_metrics
Xmetrics
*	variables
Ynon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
(:&@
2after_softmax_1/kernel
": 
2after_softmax_1/bias
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
А

Zlayers
.regularization_losses
[layer_regularization_losses
/trainable_variables
\layer_metrics
]metrics
0	variables
^non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
_0
`1"
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
Л
	atotal
	bcount
c	variables
d	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
џ
	etotal
	fcount
g
_fn_kwargs
h	variables
i	keras_api"И
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
.
a0
b1"
trackable_list_wrapper
-
c	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
e0
f1"
trackable_list_wrapper
-
h	variables"
_generic_user_object
-:+2Adam/conv1_1/kernel/m
:2Adam/conv1_1/bias/m
-:+2Adam/conv2_1/kernel/m
:2Adam/conv2_1/bias/m
.:,@2Adam/before_softmax_1/kernel/m
(:&@2Adam/before_softmax_1/bias/m
-:+@
2Adam/after_softmax_1/kernel/m
':%
2Adam/after_softmax_1/bias/m
-:+2Adam/conv1_1/kernel/v
:2Adam/conv1_1/bias/v
-:+2Adam/conv2_1/kernel/v
:2Adam/conv2_1/bias/v
.:,@2Adam/before_softmax_1/kernel/v
(:&@2Adam/before_softmax_1/bias/v
-:+@
2Adam/after_softmax_1/kernel/v
':%
2Adam/after_softmax_1/bias/v
ў2ћ
,__inference_sequential_1_layer_call_fn_98967
,__inference_sequential_1_layer_call_fn_99089
,__inference_sequential_1_layer_call_fn_98919
,__inference_sequential_1_layer_call_fn_99110Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
ф2с
 __inference__wrapped_model_98691М
В
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
annotationsЊ *,Ђ)
'$
inputџџџџџџџџџ
ъ2ч
G__inference_sequential_1_layer_call_and_return_conditional_losses_98843
G__inference_sequential_1_layer_call_and_return_conditional_losses_99033
G__inference_sequential_1_layer_call_and_return_conditional_losses_99068
G__inference_sequential_1_layer_call_and_return_conditional_losses_98870Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
2
%__inference_conv1_layer_call_fn_98713з
В
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
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
@__inference_conv1_layer_call_and_return_conditional_losses_98703з
В
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
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
#__inference_mp1_layer_call_fn_98725р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
І2Ѓ
>__inference_mp1_layer_call_and_return_conditional_losses_98719р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
%__inference_conv2_layer_call_fn_98746з
В
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
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
@__inference_conv2_layer_call_and_return_conditional_losses_98736з
В
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
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
#__inference_mp2_layer_call_fn_98758р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
І2Ѓ
>__inference_mp2_layer_call_and_return_conditional_losses_98752р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
б2Ю
'__inference_flatten_layer_call_fn_99121Ђ
В
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
annotationsЊ *
 
ь2щ
B__inference_flatten_layer_call_and_return_conditional_losses_99116Ђ
В
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
annotationsЊ *
 
и2е
.__inference_before_softmax_layer_call_fn_99141Ђ
В
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
annotationsЊ *
 
ѓ2№
I__inference_before_softmax_layer_call_and_return_conditional_losses_99132Ђ
В
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
annotationsЊ *
 
з2д
-__inference_after_softmax_layer_call_fn_99161Ђ
В
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
annotationsЊ *
 
ђ2я
H__inference_after_softmax_layer_call_and_return_conditional_losses_99152Ђ
В
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
annotationsЊ *
 
0B.
#__inference_signature_wrapper_98998inputІ
 __inference__wrapped_model_98691&',-6Ђ3
,Ђ)
'$
inputџџџџџџџџџ
Њ "=Њ:
8
after_softmax'$
after_softmaxџџџџџџџџџ
Ј
H__inference_after_softmax_layer_call_and_return_conditional_losses_99152\,-/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ

 
-__inference_after_softmax_layer_call_fn_99161O,-/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ
Љ
I__inference_before_softmax_layer_call_and_return_conditional_losses_99132\&'/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ@
 
.__inference_before_softmax_layer_call_fn_99141O&'/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџ@е
@__inference_conv1_layer_call_and_return_conditional_losses_98703IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ­
%__inference_conv1_layer_call_fn_98713IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџе
@__inference_conv2_layer_call_and_return_conditional_losses_98736IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ­
%__inference_conv2_layer_call_fn_98746IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџІ
B__inference_flatten_layer_call_and_return_conditional_losses_99116`7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 ~
'__inference_flatten_layer_call_fn_99121S7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "џџџџџџџџџс
>__inference_mp1_layer_call_and_return_conditional_losses_98719RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Й
#__inference_mp1_layer_call_fn_98725RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџс
>__inference_mp2_layer_call_and_return_conditional_losses_98752RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Й
#__inference_mp2_layer_call_fn_98758RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџМ
G__inference_sequential_1_layer_call_and_return_conditional_losses_98843q&',->Ђ;
4Ђ1
'$
inputџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ

 М
G__inference_sequential_1_layer_call_and_return_conditional_losses_98870q&',->Ђ;
4Ђ1
'$
inputџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ

 Н
G__inference_sequential_1_layer_call_and_return_conditional_losses_99033r&',-?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ

 Н
G__inference_sequential_1_layer_call_and_return_conditional_losses_99068r&',-?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ

 
,__inference_sequential_1_layer_call_fn_98919d&',->Ђ;
4Ђ1
'$
inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџ

,__inference_sequential_1_layer_call_fn_98967d&',->Ђ;
4Ђ1
'$
inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ

,__inference_sequential_1_layer_call_fn_99089e&',-?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџ

,__inference_sequential_1_layer_call_fn_99110e&',-?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
В
#__inference_signature_wrapper_98998&',-?Ђ<
Ђ 
5Њ2
0
input'$
inputџџџџџџџџџ"=Њ:
8
after_softmax'$
after_softmaxџџџџџџџџџ
