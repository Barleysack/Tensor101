  *	??????@2q
:Iterator::Model::MaxIntraOpParallelism::Prefetch::Map::MapaTR'??@!X?25%kT@)???B?i@1q???s?P@:Preprocessing2l
5Iterator::Model::MaxIntraOpParallelism::Prefetch::Maph"lxze@!?%C??vW@)??&S??1e???K[(@:Preprocessing2?
\Iterator::Model::MaxIntraOpParallelism::Prefetch::Map::Map::BatchV2::ForeverRepeat::Prefetch ?V-??!ز?6|@)?V-??1ز?6|@:Preprocessing2?
RIterator::Model::MaxIntraOpParallelism::Prefetch::Map::Map::BatchV2::ForeverRepeat ?c?ZB??!îN|?%@)??0?*??12Ӱf|J@:Preprocessing2z
CIterator::Model::MaxIntraOpParallelism::Prefetch::Map::Map::BatchV2?):????!>?%Z?U-@)
h"lxz??1???-D?@:Preprocessing2?
aIterator::Model::MaxIntraOpParallelism::Prefetch::Map::Map::BatchV2::ForeverRepeat::Prefetch::Map,?A?f????!?hń7@@)??+e???1?)???@:Preprocessing2?
nIterator::Model::MaxIntraOpParallelism::Prefetch::Map::Map::BatchV2::ForeverRepeat::Prefetch::Map::MemoryCache,!?rh????!g?aZ??@)??m4????1}zF????:Preprocessing2?
rIterator::Model::MaxIntraOpParallelism::Prefetch::Map::Map::BatchV2::ForeverRepeat::Prefetch::Map::MemoryCacheImpl,??	h"l??!QzIn`y??)??	h"l??1QzIn`y??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchΈ?????!????WA??)Έ?????1????WA??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismS?!?uq??!???????)	?^)ː?1??o???:Preprocessing2F
Iterator::Model}??b٭?!?? ?u[??)U???N@s?1?a^???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.