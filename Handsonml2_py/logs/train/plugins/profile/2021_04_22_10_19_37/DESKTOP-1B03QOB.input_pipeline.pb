  *	fffffY?@2q
:Iterator::Model::MaxIntraOpParallelism::Prefetch::Map::Map???K7?@!??Z??T@)F%u @1mEΊזQ@:Preprocessing2l
5Iterator::Model::MaxIntraOpParallelism::Prefetch::MapQk?w?"@!IG?WW@)	?^)???1?y????&@:Preprocessing2z
CIterator::Model::MaxIntraOpParallelism::Prefetch::Map::Map::BatchV2(??y??!???FF%@)t??????1	? ???@:Preprocessing2?
\Iterator::Model::MaxIntraOpParallelism::Prefetch::Map::Map::BatchV2::ForeverRepeat::Prefetch ?G?z???!uL`?'?	@)?G?z???1uL`?'?	@:Preprocessing2?
RIterator::Model::MaxIntraOpParallelism::Prefetch::Map::Map::BatchV2::ForeverRepeat ?h o???!ߛ?A??@)	?c?Z??1H봹@:Preprocessing2?
aIterator::Model::MaxIntraOpParallelism::Prefetch::Map::Map::BatchV2::ForeverRepeat::Prefetch::Map$?/?'??!~?v???@)??K7?A??1\%,?@:Preprocessing2?
nIterator::Model::MaxIntraOpParallelism::Prefetch::Map::Map::BatchV2::ForeverRepeat::Prefetch::Map::MemoryCache$1?*?Թ?!???ʄ7@)Gx$(??1˛?óx @:Preprocessing2?
rIterator::Model::MaxIntraOpParallelism::Prefetch::Map::Map::BatchV2::ForeverRepeat::Prefetch::Map::MemoryCacheImpl$/?$???!???}??)/?$???1???}??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchHP?sע?!	7????)HP?sע?1	7????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism ?o_Ω?!?\0??)_?Qڋ?1?#?*m??:Preprocessing2F
Iterator::Model_?Qګ?!?#?*m??)????Mbp?1v???????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.