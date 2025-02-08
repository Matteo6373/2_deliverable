The code was tested on the cluster. To run the code, simply copy the folder to the cluster and execute the file hello.pbs from within the copied folder. 
This will recompile and run the matrix_transposition.c code for each pair of matrix size and number of threads. Running all combinations takes about 2 minutes. To execute the code with defined 
size and n_threads, first load the gcc91 and mpich-3.2.1--gcc-9.1.0 module, then compile the code with mpicxx -o mt.out matrix_transposition.cpp -O2, and finally run it with mpirun -np (n_process) ./mt.out (size), e.g.,mpirun -np 4 ./mt.out 1024

compiler version:
g++-9.1.0 (GCC) 9.1.0
Copyright (C) 2019 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.


architecture:
The running nodes are hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it hpc-c12-node32.unitn.it
The running nodes are hpc-c12-node32.unitn.it
Node: hpc-c12-node32.unitn.it
Architecture:          x86_64
CPU op-mode(s):        32-bit, 64-bit
Byte Order:            Little Endian
CPU(s):                96
On-line CPU(s) list:   0-95
Thread(s) per core:    1
Core(s) per socket:    24
Socket(s):             4
NUMA node(s):          4
Vendor ID:             GenuineIntel
CPU family:            6
Model:                 85
Model name:            Intel(R) Xeon(R) Gold 6252N CPU @ 2.30GHz
Stepping:              7
CPU MHz:               2300.000
BogoMIPS:              4600.00
Virtualization:        VT-x
L1d cache:             32K
L1i cache:             32K
L2 cache:              1024K
L3 cache:              36608K
NUMA node0 CPU(s):     0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92
NUMA node1 CPU(s):     1,5,9,13,17,21,25,29,33,37,41,45,49,53,57,61,65,69,73,77,81,85,89,93
NUMA node2 CPU(s):     2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,66,70,74,78,82,86,90,94
NUMA node3 CPU(s):     3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63,67,71,75,79,83,87,91,95
Flags:                 fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf eagerfpu pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch epb cat_l3 cdp_l3 invpcid_single intel_ppin ssbd mba rsb_ctxsw ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm cqm mpx rdt_a avx512f avx512dq rdseed adx smap clflushopt clwb intel_pt avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local dtherm ida arat pln pts pku ospke avx512_vnni md_clear spec_ctrl intel_stibp flush_l1d arch_capabilities
process= 1
