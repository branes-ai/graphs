��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
�
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	"
grad_abool( "
grad_bbool( 
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
9
VarIsInitializedOp
resource
is_initialized
�"serve*2.18.02v2.18.0-rc2-4-g6550e4bd8028�`
�
one_layer_mlp/biasVarHandleOp*
_output_shapes
: *#

debug_nameone_layer_mlp/bias/*
dtype0*
shape:*#
shared_nameone_layer_mlp/bias
u
&one_layer_mlp/bias/Read/ReadVariableOpReadVariableOpone_layer_mlp/bias*
_output_shapes
:*
dtype0
�
#Variable/Initializer/ReadVariableOpReadVariableOpone_layer_mlp/bias*
_class
loc:@Variable*
_output_shapes
:*
dtype0
�
VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *

debug_name	Variable/*
dtype0*
shape:*
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 
_
Variable/AssignAssignVariableOpVariable#Variable/Initializer/ReadVariableOp*
dtype0
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:*
dtype0
�
one_layer_mlp/kernelVarHandleOp*
_output_shapes
: *%

debug_nameone_layer_mlp/kernel/*
dtype0*
shape
:*%
shared_nameone_layer_mlp/kernel
}
(one_layer_mlp/kernel/Read/ReadVariableOpReadVariableOpone_layer_mlp/kernel*
_output_shapes

:*
dtype0
�
%Variable_1/Initializer/ReadVariableOpReadVariableOpone_layer_mlp/kernel*
_class
loc:@Variable_1*
_output_shapes

:*
dtype0
�

Variable_1VarHandleOp*
_class
loc:@Variable_1*
_output_shapes
: *

debug_nameVariable_1/*
dtype0*
shape
:*
shared_name
Variable_1
e
+Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_1*
_output_shapes
: 
e
Variable_1/AssignAssignVariableOp
Variable_1%Variable_1/Initializer/ReadVariableOp*
dtype0
i
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes

:*
dtype0
g
serving_default_inputsPlaceholder*
_output_shapes

:
*
dtype0*
shape
:

�
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputsone_layer_mlp/kernelone_layer_mlp/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU 2J 8� �J *9
f4R2
0__inference_signature_wrapper_serving_default_86

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
fc
_default_save_signature
_inbound_nodes
_outbound_nodes
_losses
	_loss_ids
_losses_override
_build_shapes_dict
	
signatures*
�

_kernel
bias
_inbound_nodes
_outbound_nodes
_losses
	_loss_ids
_losses_override
_build_shapes_dict*

trace_0* 
* 
* 
* 
* 
* 
* 

serving_default* 
IC
VARIABLE_VALUE
Variable_1%fc/_kernel/.ATTRIBUTES/VARIABLE_VALUE*
D>
VARIABLE_VALUEVariable"fc/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename
Variable_1VariableConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU 2J 8� �J *%
f R
__inference__traced_save_128
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename
Variable_1Variable*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU 2J 8� �J *(
f#R!
__inference__traced_restore_143�C
�
�
__inference__traced_restore_143
file_prefix-
assignvariableop_variable_1:)
assignvariableop_1_variable:

identity_3��AssignVariableOp�AssignVariableOp_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*|
valuesBqB%fc/_kernel/.ATTRIBUTES/VARIABLE_VALUEB"fc/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHv
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0* 
_output_shapes
:::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_variable_1Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_variableIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_2Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_3IdentityIdentity_2:output:0^NoOp_1*
T0*
_output_shapes
: L
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2(
AssignVariableOp_1AssignVariableOp_12$
AssignVariableOpAssignVariableOp:($
"
_user_specified_name
Variable:*&
$
_user_specified_name
Variable_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
0__inference_signature_wrapper_serving_default_86

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU 2J 8� �J *'
f"R 
__inference_serving_default_76f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:
: : 22
StatefulPartitionedCallStatefulPartitionedCall:"

_user_specified_name82:"

_user_specified_name80:F B

_output_shapes

:

 
_user_specified_nameinputs
�
�
__inference_serving_default_76

inputsF
4one_layer_mlp_1_dense_1_cast_readvariableop_resource:E
7one_layer_mlp_1_dense_1_biasadd_readvariableop_resource:
identity��.one_layer_mlp_1/dense_1/BiasAdd/ReadVariableOp�+one_layer_mlp_1/dense_1/Cast/ReadVariableOp�
+one_layer_mlp_1/dense_1/Cast/ReadVariableOpReadVariableOp4one_layer_mlp_1_dense_1_cast_readvariableop_resource*
_output_shapes

:*
dtype0�
one_layer_mlp_1/dense_1/MatMulMatMulinputs3one_layer_mlp_1/dense_1/Cast/ReadVariableOp:value:0*
T0*
_output_shapes

:
�
.one_layer_mlp_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp7one_layer_mlp_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
one_layer_mlp_1/dense_1/BiasAddBiasAdd(one_layer_mlp_1/dense_1/MatMul:product:06one_layer_mlp_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:
}
one_layer_mlp_1/dense_1/SoftmaxSoftmax(one_layer_mlp_1/dense_1/BiasAdd:output:0*
T0*
_output_shapes

:
o
IdentityIdentity)one_layer_mlp_1/dense_1/Softmax:softmax:0^NoOp*
T0*
_output_shapes

:
�
NoOpNoOp/^one_layer_mlp_1/dense_1/BiasAdd/ReadVariableOp,^one_layer_mlp_1/dense_1/Cast/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:
: : 2`
.one_layer_mlp_1/dense_1/BiasAdd/ReadVariableOp.one_layer_mlp_1/dense_1/BiasAdd/ReadVariableOp2Z
+one_layer_mlp_1/dense_1/Cast/ReadVariableOp+one_layer_mlp_1/dense_1/Cast/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:F B

_output_shapes

:

 
_user_specified_nameinputs
�
�
__inference__traced_save_128
file_prefix3
!read_disablecopyonread_variable_1:/
!read_1_disablecopyonread_variable:
savev2_const

identity_5��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: d
Read/DisableCopyOnReadDisableCopyOnRead!read_disablecopyonread_variable_1*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp!read_disablecopyonread_variable_1^Read/DisableCopyOnRead*
_output_shapes

:*
dtype0Z
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:f
Read_1/DisableCopyOnReadDisableCopyOnRead!read_1_disablecopyonread_variable*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp!read_1_disablecopyonread_variable^Read_1/DisableCopyOnRead*
_output_shapes
:*
dtype0Z

Identity_2IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*|
valuesBqB%fc/_kernel/.ATTRIBUTES/VARIABLE_VALUEB"fc/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHs
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 h

Identity_4Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: S

Identity_5IdentityIdentity_4:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp*
_output_shapes
 "!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:($
"
_user_specified_name
Variable:*&
$
_user_specified_name
Variable_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
0
inputs&
serving_default_inputs:0
3
output_0'
StatefulPartitionedCall:0
tensorflow/serving/predict:�
�
fc
_default_save_signature
_inbound_nodes
_outbound_nodes
_losses
	_loss_ids
_losses_override
_build_shapes_dict
	
signatures"
_generic_user_object
�

_kernel
bias
_inbound_nodes
_outbound_nodes
_losses
	_loss_ids
_losses_override
_build_shapes_dict"
_generic_user_object
�
trace_02�
__inference_serving_default_76�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�
�
ztrace_0
 "
trackable_list_wrapper
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
,
serving_default"
signature_map
&:$2one_layer_mlp/kernel
 :2one_layer_mlp/bias
 "
trackable_list_wrapper
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
�B�
__inference_serving_default_76inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
0__inference_signature_wrapper_serving_default_86inputs"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�

jinputs
kwonlydefaults
 
annotations� *
 h
__inference_serving_default_76F
&�#
�
�
inputs

� "�
unknown
�
0__inference_signature_wrapper_serving_default_86b
0�-
� 
&�#
!
inputs�
inputs
"*�'
%
output_0�
output_0
