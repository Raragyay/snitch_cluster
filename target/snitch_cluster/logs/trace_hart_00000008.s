#; unknown function (??:0)
#; 
             36000    0x00001000 csrr    a0, mhartid            #; mhartid = 8, (wrb) a0  <-- 8
             60000    0x00001004 auipc   a1, 0x0                #; (wrb) a1  <-- 4100
             84000    0x00001008 addi    a1, a1, 32             #; a1  = 4100, (wrb) a1  <-- 4132
            108000    0x0000100c auipc   t0, 0x0                #; (wrb) t0  <-- 4108
            132000    0x00001010 lw      t0, 20(t0)             #; t0  = 4108, t0  <~~ Word[0x00001020]
            162000                                              #; (lsu) t0  <-- 0x80000000
            172000    0x00001014 jalr    t0                     #; t0  = 0x80000000, (wrb) ra  <-- 4120, goto 0x80000000
#; unknown function (start.S:12)
#;   mv t0, x0
            175000    0x80000000 li      t0, 0                  #; (wrb) t0  <-- 0
#; unknown function (start.S:13)
#;   mv t1, x0
            176000    0x80000004 li      t1, 0                  #; (wrb) t1  <-- 0
#; unknown function (start.S:14)
#;   mv t2, x0
            177000    0x80000008 li      t2, 0                  #; (wrb) t2  <-- 0
#; unknown function (start.S:15)
#;   mv t3, x0
            178000    0x8000000c li      t3, 0                  #; (wrb) t3  <-- 0
#; unknown function (start.S:16)
#;   mv t4, x0
            179000    0x80000010 li      t4, 0                  #; (wrb) t4  <-- 0
#; unknown function (start.S:17)
#;   mv t5, x0
            180000    0x80000014 li      t5, 0                  #; (wrb) t5  <-- 0
#; unknown function (start.S:18)
#;   mv t6, x0
            181000    0x80000018 li      t6, 0                  #; (wrb) t6  <-- 0
#; unknown function (start.S:19)
#;   mv a0, x0
            182000    0x8000001c li      a0, 0                  #; (wrb) a0  <-- 0
#; unknown function (start.S:20)
#;   mv a1, x0
            183000    0x80000020 li      a1, 0                  #; (wrb) a1  <-- 0
#; unknown function (start.S:21)
#;   mv a2, x0
            184000    0x80000024 li      a2, 0                  #; (wrb) a2  <-- 0
#; unknown function (start.S:22)
#;   mv a3, x0
            185000    0x80000028 li      a3, 0                  #; (wrb) a3  <-- 0
#; unknown function (start.S:23)
#;   mv a4, x0
            186000    0x8000002c li      a4, 0                  #; (wrb) a4  <-- 0
#; unknown function (start.S:24)
#;   mv a5, x0
            187000    0x80000030 li      a5, 0                  #; (wrb) a5  <-- 0
#; unknown function (start.S:25)
#;   mv a6, x0
            188000    0x80000034 li      a6, 0                  #; (wrb) a6  <-- 0
#; unknown function (start.S:26)
#;   mv a7, x0
            189000    0x80000038 li      a7, 0                  #; (wrb) a7  <-- 0
#; unknown function (start.S:27)
#;   mv s0, x0
            190000    0x8000003c li      s0, 0                  #; (wrb) s0  <-- 0
#; unknown function (start.S:28)
#;   mv s1, x0
            191000    0x80000040 li      s1, 0                  #; (wrb) s1  <-- 0
#; unknown function (start.S:29)
#;   mv s2, x0
            192000    0x80000044 li      s2, 0                  #; (wrb) s2  <-- 0
#; unknown function (start.S:30)
#;   mv s3, x0
            193000    0x80000048 li      s3, 0                  #; (wrb) s3  <-- 0
#; unknown function (start.S:31)
#;   mv s4, x0
            194000    0x8000004c li      s4, 0                  #; (wrb) s4  <-- 0
#; unknown function (start.S:32)
#;   mv s5, x0
            195000    0x80000050 li      s5, 0                  #; (wrb) s5  <-- 0
#; unknown function (start.S:33)
#;   mv s6, x0
            196000    0x80000054 li      s6, 0                  #; (wrb) s6  <-- 0
#; unknown function (start.S:34)
#;   mv s7, x0
            197000    0x80000058 li      s7, 0                  #; (wrb) s7  <-- 0
#; unknown function (start.S:35)
#;   mv s8, x0
            198000    0x8000005c li      s8, 0                  #; (wrb) s8  <-- 0
#; unknown function (start.S:36)
#;   mv s9, x0
            199000    0x80000060 li      s9, 0                  #; (wrb) s9  <-- 0
#; unknown function (start.S:37)
#;   mv s10, x0
            200000    0x80000064 li      s10, 0                 #; (wrb) s10 <-- 0
#; unknown function (start.S:38)
#;   mv s11, x0
            201000    0x80000068 li      s11, 0                 #; (wrb) s11 <-- 0
#; unknown function (start.S:44)
#;   csrr    t0, misa
            202000    0x8000006c csrr    t0, misa               #; misa = 0x40801129, (wrb) t0  <-- 0x40801129
#; unknown function (start.S:45)
#;   andi    t0, t0, (1 << 3) | (1 << 5) # D/F - single/double precision float extension
            203000    0x80000070 andi    t0, t0, 40             #; t0  = 0x40801129, (wrb) t0  <-- 40
#; unknown function (start.S:46)
#;   beqz    t0, 3f
            204000    0x80000074 beqz    t0, pc + 132           #; t0  = 40, not taken
#; unknown function (start.S:48)
#;   fcvt.d.w f0, zero
            206000    0x80000078 fcvt.d.w ft0, zero             #; ac1  = 0
#; unknown function (start.S:49)
#;   fcvt.d.w f1, zero
            207000    0x8000007c fcvt.d.w ft1, zero             #; ac1  = 0, (f:fpu) ft0  <-- 0.0
#; unknown function (start.S:50)
#;   fcvt.d.w f2, zero
            208000    0x80000080 fcvt.d.w ft2, zero             #; ac1  = 0, (f:fpu) ft1  <-- 0.0
#; unknown function (start.S:51)
#;   fcvt.d.w f3, zero
            209000    0x80000084 fcvt.d.w ft3, zero             #; ac1  = 0, (f:fpu) ft2  <-- 0.0
#; unknown function (start.S:52)
#;   fcvt.d.w f4, zero
            210000    0x80000088 fcvt.d.w ft4, zero             #; ac1  = 0, (f:fpu) ft3  <-- 0.0
#; unknown function (start.S:53)
#;   fcvt.d.w f5, zero
            211000    0x8000008c fcvt.d.w ft5, zero             #; ac1  = 0, (f:fpu) ft4  <-- 0.0
#; unknown function (start.S:54)
#;   fcvt.d.w f6, zero
            212000    0x80000090 fcvt.d.w ft6, zero             #; ac1  = 0, (f:fpu) ft5  <-- 0.0
#; unknown function (start.S:55)
#;   fcvt.d.w f7, zero
            213000    0x80000094 fcvt.d.w ft7, zero             #; ac1  = 0, (f:fpu) ft6  <-- 0.0
#; unknown function (start.S:56)
#;   fcvt.d.w f8, zero
            214000    0x80000098 fcvt.d.w fs0, zero             #; ac1  = 0, (f:fpu) ft7  <-- 0.0
#; unknown function (start.S:57)
#;   fcvt.d.w f9, zero
            215000    0x8000009c fcvt.d.w fs1, zero             #; ac1  = 0, (f:fpu) fs0  <-- 0.0
#; unknown function (start.S:58)
#;   fcvt.d.w f10, zero
            216000    0x800000a0 fcvt.d.w fa0, zero             #; ac1  = 0, (f:fpu) fs1  <-- 0.0
#; unknown function (start.S:59)
#;   fcvt.d.w f11, zero
            217000    0x800000a4 fcvt.d.w fa1, zero             #; ac1  = 0, (f:fpu) fa0  <-- 0.0
#; unknown function (start.S:60)
#;   fcvt.d.w f12, zero
            218000    0x800000a8 fcvt.d.w fa2, zero             #; ac1  = 0, (f:fpu) fa1  <-- 0.0
#; unknown function (start.S:61)
#;   fcvt.d.w f13, zero
            219000    0x800000ac fcvt.d.w fa3, zero             #; ac1  = 0, (f:fpu) fa2  <-- 0.0
#; unknown function (start.S:62)
#;   fcvt.d.w f14, zero
            220000    0x800000b0 fcvt.d.w fa4, zero             #; ac1  = 0, (f:fpu) fa3  <-- 0.0
#; unknown function (start.S:63)
#;   fcvt.d.w f15, zero
            221000    0x800000b4 fcvt.d.w fa5, zero             #; ac1  = 0, (f:fpu) fa4  <-- 0.0
#; unknown function (start.S:64)
#;   fcvt.d.w f16, zero
            222000    0x800000b8 fcvt.d.w fa6, zero             #; ac1  = 0, (f:fpu) fa5  <-- 0.0
#; unknown function (start.S:65)
#;   fcvt.d.w f17, zero
            223000    0x800000bc fcvt.d.w fa7, zero             #; ac1  = 0, (f:fpu) fa6  <-- 0.0
            224000                                              #; (f:fpu) fa7  <-- 0.0
#; unknown function (start.S:66)
#;   fcvt.d.w f18, zero
            225000    0x800000c0 fcvt.d.w fs2, zero             #; ac1  = 0
#; unknown function (start.S:67)
#;   fcvt.d.w f19, zero
            226000    0x800000c4 fcvt.d.w fs3, zero             #; ac1  = 0, (f:fpu) fs2  <-- 0.0
#; unknown function (start.S:68)
#;   fcvt.d.w f20, zero
            227000    0x800000c8 fcvt.d.w fs4, zero             #; ac1  = 0, (f:fpu) fs3  <-- 0.0
#; unknown function (start.S:69)
#;   fcvt.d.w f21, zero
            228000    0x800000cc fcvt.d.w fs5, zero             #; ac1  = 0, (f:fpu) fs4  <-- 0.0
#; unknown function (start.S:70)
#;   fcvt.d.w f22, zero
            229000    0x800000d0 fcvt.d.w fs6, zero             #; ac1  = 0, (f:fpu) fs5  <-- 0.0
#; unknown function (start.S:71)
#;   fcvt.d.w f23, zero
            230000    0x800000d4 fcvt.d.w fs7, zero             #; ac1  = 0, (f:fpu) fs6  <-- 0.0
#; unknown function (start.S:72)
#;   fcvt.d.w f24, zero
            231000    0x800000d8 fcvt.d.w fs8, zero             #; ac1  = 0, (f:fpu) fs7  <-- 0.0
#; unknown function (start.S:73)
#;   fcvt.d.w f25, zero
            232000    0x800000dc fcvt.d.w fs9, zero             #; ac1  = 0, (f:fpu) fs8  <-- 0.0
            233000                                              #; (f:fpu) fs9  <-- 0.0
#; unknown function (start.S:74)
#;   fcvt.d.w f26, zero
            237000    0x800000e0 fcvt.d.w fs10, zero            #; ac1  = 0
#; unknown function (start.S:75)
#;   fcvt.d.w f27, zero
            238000    0x800000e4 fcvt.d.w fs11, zero            #; ac1  = 0, (f:fpu) fs10 <-- 0.0
#; unknown function (start.S:76)
#;   fcvt.d.w f28, zero
            239000    0x800000e8 fcvt.d.w ft8, zero             #; ac1  = 0, (f:fpu) fs11 <-- 0.0
#; unknown function (start.S:77)
#;   fcvt.d.w f29, zero
            240000    0x800000ec fcvt.d.w ft9, zero             #; ac1  = 0, (f:fpu) ft8  <-- 0.0
#; unknown function (start.S:78)
#;   fcvt.d.w f30, zero
            241000    0x800000f0 fcvt.d.w ft10, zero            #; ac1  = 0, (f:fpu) ft9  <-- 0.0
#; unknown function (start.S:88)
#;   1:  auipc   gp, %pcrel_hi(__global_pointer$)
            242000    0x800000f8 auipc   gp, 0x7                #; (wrb) gp  <-- 0x800070f8
                 M    0x800000f4 fcvt.d.w ft11, zero            #; ac1  = 0, (f:fpu) ft10 <-- 0.0
#; unknown function (start.S:89)
#;   addi    gp, gp, %pcrel_lo(1b)
            243000    0x800000fc addi    gp, gp, 928            #; gp  = 0x800070f8, (wrb) gp  <-- 0x80007498
                 M                                              #; (f:fpu) ft11 <-- 0.0
#; unknown function (start.S:98)
#;   csrr a0, mhartid
            248000    0x80000100 csrr    a0, mhartid            #; mhartid = 8, (wrb) a0  <-- 8
#; unknown function (start.S:99)
#;   li   t0, SNRT_BASE_HARTID
            249000    0x80000104 li      t0, 0                  #; (wrb) t0  <-- 0
#; unknown function (start.S:100)
#;   sub  a0, a0, t0
            250000    0x80000108 sub     a0, a0, t0             #; a0  = 8, t0  = 0, (wrb) a0  <-- 8
#; unknown function (start.S:101)
#;   li   a1, SNRT_CLUSTER_CORE_NUM
            251000    0x8000010c li      a1, 9                  #; (wrb) a1  <-- 9
#; unknown function (start.S:102)
#;   div  t0, a0, a1
            252000    0x80000110 div     t0, a0, a1             #; a0  = 8, a1  = 9
#; unknown function (start.S:105)
#;   remu a0, a0, a1
            253000    0x80000114 remu    a0, a0, a1             #; a0  = 8, a1  = 9
#; unknown function (start.S:108)
#;   li   a2, SNRT_TCDM_START_ADDR
            254000    0x80000118 lui     a2, 0x10000            #; (wrb) a2  <-- 0x10000000
#; unknown function (start.S:109)
#;   li   t1, SNRT_CLUSTER_OFFSET
            255000    0x8000011c li      t1, 0                  #; (wrb) t1  <-- 0
            272000                                              #; (acc) t0  <-- 0
#; unknown function (start.S:110)
#;   mul  t0, t1, t0
            273000    0x80000120 mul     t0, t1, t0             #; t1  = 0, t0  = 0
            291000                                              #; (acc) a0  <-- 8
            300000                                              #; (acc) t0  <-- 0
#; unknown function (start.S:111)
#;   add  a2, a2, t0
            301000    0x80000124 add     a2, a2, t0             #; a2  = 0x10000000, t0  = 0, (wrb) a2  <-- 0x10000000
#; unknown function (start.S:114)
#;   li   t0, SNRT_TCDM_SIZE
            302000    0x80000128 lui     t0, 0x20               #; (wrb) t0  <-- 0x00020000
#; unknown function (start.S:115)
#;   add  a2, a2, t0
            303000    0x8000012c add     a2, a2, t0             #; a2  = 0x10000000, t0  = 0x00020000, (wrb) a2  <-- 0x10020000
#; unknown function (start.S:121)
#;   la        t0, __cdata_end
            304000    0x80000130 auipc   t0, 0x7                #; (wrb) t0  <-- 0x80007130
            305000    0x80000134 addi    t0, t0, -1176          #; t0  = 0x80007130, (wrb) t0  <-- 0x80006c98
#; unknown function (start.S:122)
#;   la        t1, __cdata_start
            306000    0x80000138 auipc   t1, 0x7                #; (wrb) t1  <-- 0x80007138
            307000    0x8000013c addi    t1, t1, -1184          #; t1  = 0x80007138, (wrb) t1  <-- 0x80006c98
#; unknown function (start.S:123)
#;   sub       t0, t0, t1
            308000    0x80000140 sub     t0, t0, t1             #; t0  = 0x80006c98, t1  = 0x80006c98, (wrb) t0  <-- 0
#; unknown function (start.S:124)
#;   sub       a2, a2, t0
            309000    0x80000144 sub     a2, a2, t0             #; a2  = 0x10020000, t0  = 0, (wrb) a2  <-- 0x10020000
#; unknown function (start.S:125)
#;   la        t0, __cbss_end
            310000    0x80000148 auipc   t0, 0x7                #; (wrb) t0  <-- 0x80007148
            311000    0x8000014c addi    t0, t0, -1184          #; t0  = 0x80007148, (wrb) t0  <-- 0x80006ca8
#; unknown function (start.S:126)
#;   la        t1, __cbss_start
            312000    0x80000150 auipc   t1, 0x7                #; (wrb) t1  <-- 0x80007150
            313000    0x80000154 addi    t1, t1, -1208          #; t1  = 0x80007150, (wrb) t1  <-- 0x80006c98
#; unknown function (start.S:127)
#;   sub       t0, t0, t1
            314000    0x80000158 sub     t0, t0, t1             #; t0  = 0x80006ca8, t1  = 0x80006c98, (wrb) t0  <-- 16
#; unknown function (start.S:128)
#;   sub       a2, a2, t0
            315000    0x8000015c sub     a2, a2, t0             #; a2  = 0x10020000, t0  = 16, (wrb) a2  <-- 0x1001fff0
#; unknown function (start.S:135)
#;   addi      a2, a2, -8
            316000    0x80000160 addi    a2, a2, -8             #; a2  = 0x1001fff0, (wrb) a2  <-- 0x1001ffe8
#; unknown function (start.S:136)
#;   sw        zero, 0(a2)
            317000    0x80000164 sw      zero, 0(a2)            #; a2  = 0x1001ffe8, 0 ~~> Word[0x1001ffe8]
#; unknown function (start.S:140)
#;   sll       t0, a0, SNRT_LOG2_STACK_SIZE
            318000    0x80000168 slli    t0, a0, 10             #; a0  = 8, (wrb) t0  <-- 8192
#; unknown function (start.S:143)
#;   sub       sp, a2, t0
            319000    0x8000016c sub     sp, a2, t0             #; a2  = 0x1001ffe8, t0  = 8192, (wrb) sp  <-- 0x1001dfe8
#; unknown function (start.S:147)
#;   sll       t1, a1, t2
            320000    0x80000170 sll     t1, a1, t2             #; a1  = 9, t2  = 0, (wrb) t1  <-- 9
#; unknown function (start.S:148)
#;   sub       a2, a2, t1
            321000    0x80000174 sub     a2, a2, t1             #; a2  = 0x1001ffe8, t1  = 9, (wrb) a2  <-- 0x1001ffdf
#; unknown function (start.S:151)
#;   slli      t0, a0, 3  # this hart
            322000    0x80000178 slli    t0, a0, 3              #; a0  = 8, (wrb) t0  <-- 64
#; unknown function (start.S:152)
#;   slli      t1, a1, 3  # all harts
            323000    0x8000017c slli    t1, a1, 3              #; a1  = 9, (wrb) t1  <-- 72
#; unknown function (start.S:153)
#;   sub       sp, sp, t0
            324000    0x80000180 sub     sp, sp, t0             #; sp  = 0x1001dfe8, t0  = 64, (wrb) sp  <-- 0x1001dfa8
#; unknown function (start.S:154)
#;   sub       a2, a2, t1
            325000    0x80000184 sub     a2, a2, t1             #; a2  = 0x1001ffdf, t1  = 72, (wrb) a2  <-- 0x1001ff97
#; unknown function (start.S:160)
#;   la        t0, __tdata_end
            326000    0x80000188 auipc   t0, 0x7                #; (wrb) t0  <-- 0x80007188
            327000    0x8000018c addi    t0, t0, -1280          #; t0  = 0x80007188, (wrb) t0  <-- 0x80006c88
#; unknown function (start.S:161)
#;   la        t1, __tdata_start
            328000    0x80000190 auipc   t1, 0x7                #; (wrb) t1  <-- 0x80007190
            329000    0x80000194 addi    t1, t1, -1288          #; t1  = 0x80007190, (wrb) t1  <-- 0x80006c88
#; unknown function (start.S:162)
#;   sub       t0, t0, t1
            330000    0x80000198 sub     t0, t0, t1             #; t0  = 0x80006c88, t1  = 0x80006c88, (wrb) t0  <-- 0
#; unknown function (start.S:163)
#;   sub       sp, sp, t0
            331000    0x8000019c sub     sp, sp, t0             #; sp  = 0x1001dfa8, t0  = 0, (wrb) sp  <-- 0x1001dfa8
#; unknown function (start.S:164)
#;   la        t0, __tbss_end
            336000    0x800001a0 auipc   t0, 0x7                #; (wrb) t0  <-- 0x800071a0
            337000    0x800001a4 addi    t0, t0, -1288          #; t0  = 0x800071a0, (wrb) t0  <-- 0x80006c98
#; unknown function (start.S:165)
#;   la        t1, __tbss_start
            338000    0x800001a8 auipc   t1, 0x7                #; (wrb) t1  <-- 0x800071a8
            339000    0x800001ac addi    t1, t1, -1312          #; t1  = 0x800071a8, (wrb) t1  <-- 0x80006c88
#; unknown function (start.S:166)
#;   sub       t0, t0, t1
            340000    0x800001b0 sub     t0, t0, t1             #; t0  = 0x80006c98, t1  = 0x80006c88, (wrb) t0  <-- 16
#; unknown function (start.S:167)
#;   sub       sp, sp, t0
            341000    0x800001b4 sub     sp, sp, t0             #; sp  = 0x1001dfa8, t0  = 16, (wrb) sp  <-- 0x1001df98
#; unknown function (start.S:168)
#;   andi      sp, sp, ~0x7 # align to 8B
            342000    0x800001b8 andi    sp, sp, -8             #; sp  = 0x1001df98, (wrb) sp  <-- 0x1001df98
#; unknown function (start.S:170)
#;   mv        tp, sp
            343000    0x800001bc mv      tp, sp                 #; sp  = 0x1001df98, (wrb) tp  <-- 0x1001df98
#; unknown function (start.S:172)
#;   andi      sp, sp, ~0x7 # align stack to 8B
            348000    0x800001c0 andi    sp, sp, -8             #; sp  = 0x1001df98, (wrb) sp  <-- 0x1001df98
#; unknown function (start.S:183)
#;   call snrt_main
            349000    0x800001c4 auipc   ra, 0x6                #; (wrb) ra  <-- 0x800061c4
            350000    0x800001c8 jalr    ra, ra, -796           #; ra  = 0x800061c4, (wrb) ra  <-- 0x800001cc, goto 0x80005ea8
#; snrt_main (start.c:106)
#;   void snrt_main() {
            362000    0x80005ea8 addi    sp, sp, -32            #; sp  = 0x1001df98, (wrb) sp  <-- 0x1001df78
#; snrt_main (start.c:-1)
#; 
            363000    0x80005eac sw      ra, 28(sp)             #; sp  = 0x1001df78, 0x800001cc ~~> Word[0x1001df94]
            364000    0x80005eb0 sw      s0, 24(sp)             #; sp  = 0x1001df78, 0 ~~> Word[0x1001df90]
            365000    0x80005eb4 sw      s1, 20(sp)             #; sp  = 0x1001df78, 0 ~~> Word[0x1001df8c]
            366000    0x80005eb8 sw      s2, 16(sp)             #; sp  = 0x1001df78, 0 ~~> Word[0x1001df88]
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:29)
#;     snrt_is_dm_core (team.h:57)
#;       snrt_is_compute_core (team.h:53)
#;         snrt_cluster_core_idx (team.h:41)
#;           snrt_global_core_idx (team.h:28)
#;             snrt_hartid (team.h:7)
#;               asm("csrr %0, mhartid" : "=r"(hartid));
            367000    0x80005ebc csrr    s2, mhartid            #; mhartid = 8, (wrb) s2  <-- 8
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:29)
#;     snrt_is_dm_core (team.h:57)
#;       snrt_is_compute_core (team.h:53)
#;         snrt_cluster_core_idx (team.h:41)
#;           snrt_global_core_idx (team.h:28)
#;             snrt_hartid (team.h:-1)
#; 
            374000    0x80005ec0 lui     a0, 0x38e39            #; (wrb) a0  <-- 0x38e39000
            375000    0x80005ec4 addi    a0, a0, -455           #; a0  = 0x38e39000, (wrb) a0  <-- 0x38e38e39
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:29)
#;     snrt_is_dm_core (team.h:57)
#;       snrt_is_compute_core (team.h:53)
#;         snrt_cluster_core_idx (team.h:41)
#;           return snrt_global_core_idx() % snrt_cluster_core_num();
            376000    0x80005ec8 mulhu   a0, s2, a0             #; s2  = 8, a0  = 0x38e38e39
            379000                                              #; (acc) a0  <-- 1
            380000    0x80005ecc srli    a0, a0, 1              #; a0  = 1, (wrb) a0  <-- 0
            381000    0x80005ed0 slli    a1, a0, 3              #; a0  = 0, (wrb) a1  <-- 0
            382000    0x80005ed4 add     a0, a1, a0             #; a1  = 0, a0  = 0, (wrb) a0  <-- 0
            383000    0x80005ed8 sub     a5, s2, a0             #; s2  = 8, a0  = 0, (wrb) a5  <-- 8
            384000    0x80005edc li      a6, 8                  #; (wrb) a6  <-- 8
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:29)
#;     if (snrt_is_dm_core()) {
            386000    0x80005ee0 bltu    a5, a6, pc + 524       #; a5  = 8, a6  = 8, not taken
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:33)
#;     asm volatile("mv %0, tp" : "=r"(tls_ptr) : :);
            387000    0x80005ee4 mv      a0, tp                 #; tp  = 0x1001df98, (wrb) a0  <-- 0x1001df98
            388000    0x80005ee8 sw      a0, 12(sp)             #; sp  = 0x1001df78, 0x1001df98 ~~> Word[0x1001df84]
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:34)
#;     snrt_dma_start_1d((void*)tls_ptr, (void*)(&__tdata_start), size);
            389000    0x80005eec lw      a0, 12(sp)             #; sp  = 0x1001df78, a0  <~~ Word[0x1001df84]
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:34)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:13)
#;         if (size > 0) {
            390000    0x80005ef0 auipc   a2, 0x1                #; (wrb) a2  <-- 0x80006ef0
            391000    0x80005ef4 addi    a2, a2, -616           #; a2  = 0x80006ef0, (wrb) a2  <-- 0x80006c88
            392000    0x80005ef8 auipc   a1, 0x1                #; (wrb) a1  <-- 0x80006ef8
            393000    0x80005efc addi    a1, a1, -624           #; a1  = 0x80006ef8, (wrb) a1  <-- 0x80006c88
            394000                                              #; (lsu) a0  <-- 0x1001df98
            398000    0x80005f00 sub     a4, a1, a2             #; a1  = 0x80006c88, a2  = 0x80006c88, (wrb) a4  <-- 0
            399000    0x80005f04 beqz    a4, pc + 24            #; a4  = 0, taken, goto 0x80005f1c
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:41)
#;     snrt_dma_start_1d((void*)(tls_ptr + i * tls_offset), (void*)tls_ptr,
            400000    0x80005f1c lw      a0, 12(sp)             #; sp  = 0x1001df78, a0  <~~ Word[0x1001df84]
            403000                                              #; (lsu) a0  <-- 0x1001df98
            418000    0x80005f20 lw      a2, 12(sp)             #; sp  = 0x1001df78, a2  <~~ Word[0x1001df84]
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:41)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:13)
#;         if (size > 0) {
            419000    0x80005f24 beqz    a4, pc + 28            #; a4  = 0, taken, goto 0x80005f40
            421000                                              #; (lsu) a2  <-- 0x1001df98
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:41)
#;     snrt_dma_start_1d((void*)(tls_ptr + i * tls_offset), (void*)tls_ptr,
            430000    0x80005f40 lw      a0, 12(sp)             #; sp  = 0x1001df78, a0  <~~ Word[0x1001df84]
            433000                                              #; (lsu) a0  <-- 0x1001df98
            434000    0x80005f44 lw      a2, 12(sp)             #; sp  = 0x1001df78, a2  <~~ Word[0x1001df84]
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:-1)
#; 
            435000    0x80005f48 lui     a1, 0x1                #; (wrb) a1  <-- 4096
            436000    0x80005f4c addi    a7, a1, -2032          #; a1  = 4096, (wrb) a7  <-- 2064
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:41)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:13)
#;         if (size > 0) {
            437000    0x80005f50 beqz    a4, pc + 28            #; a4  = 0, taken, (lsu) a2  <-- 0x1001df98, goto 0x80005f6c
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:41)
#;     snrt_dma_start_1d((void*)(tls_ptr + i * tls_offset), (void*)tls_ptr,
            442000    0x80005f6c lw      a0, 12(sp)             #; sp  = 0x1001df78, a0  <~~ Word[0x1001df84]
            445000                                              #; (lsu) a0  <-- 0x1001df98
            446000    0x80005f70 lw      a2, 12(sp)             #; sp  = 0x1001df78, a2  <~~ Word[0x1001df84]
            447000    0x80005f74 beqz    a4, pc + 32            #; a4  = 0, taken, goto 0x80005f94
            449000                                              #; (lsu) a2  <-- 0x1001df98
            454000    0x80005f94 lw      a0, 12(sp)             #; sp  = 0x1001df78, a0  <~~ Word[0x1001df84]
            457000                                              #; (lsu) a0  <-- 0x1001df98
            458000    0x80005f98 lw      a2, 12(sp)             #; sp  = 0x1001df78, a2  <~~ Word[0x1001df84]
            459000    0x80005f9c lui     a1, 0x1                #; (wrb) a1  <-- 4096
            461000                                              #; (lsu) a2  <-- 0x1001df98
            466000    0x80005fa0 addi    t0, a1, 32             #; a1  = 4096, (wrb) t0  <-- 4128
            467000    0x80005fa4 beqz    a4, pc + 28            #; a4  = 0, taken, goto 0x80005fc0
            478000    0x80005fc0 lw      a0, 12(sp)             #; sp  = 0x1001df78, a0  <~~ Word[0x1001df84]
            481000                                              #; (lsu) a0  <-- 0x1001df98
            482000    0x80005fc4 lw      a2, 12(sp)             #; sp  = 0x1001df78, a2  <~~ Word[0x1001df84]
            483000    0x80005fc8 beqz    a4, pc + 32            #; a4  = 0, taken, goto 0x80005fe8
            485000                                              #; (lsu) a2  <-- 0x1001df98
            490000    0x80005fe8 lw      a0, 12(sp)             #; sp  = 0x1001df78, a0  <~~ Word[0x1001df84]
            493000                                              #; (lsu) a0  <-- 0x1001df98
            494000    0x80005fec lw      a2, 12(sp)             #; sp  = 0x1001df78, a2  <~~ Word[0x1001df84]
            495000    0x80005ff0 lui     a1, 0x2                #; (wrb) a1  <-- 8192
            496000    0x80005ff4 addi    t1, a1, -2000          #; a1  = 8192, (wrb) t1  <-- 6192
            497000    0x80005ff8 beqz    a4, pc + 28            #; a4  = 0, taken, (lsu) a2  <-- 0x1001df98, goto 0x80006014
            502000    0x80006014 lw      a0, 12(sp)             #; sp  = 0x1001df78, a0  <~~ Word[0x1001df84]
            505000                                              #; (lsu) a0  <-- 0x1001df98
            506000    0x80006018 lw      a2, 12(sp)             #; sp  = 0x1001df78, a2  <~~ Word[0x1001df84]
            507000    0x8000601c beqz    a4, pc + 32            #; a4  = 0, taken, goto 0x8000603c
            509000                                              #; (lsu) a2  <-- 0x1001df98
            514000    0x8000603c lw      a0, 12(sp)             #; sp  = 0x1001df78, a0  <~~ Word[0x1001df84]
            517000                                              #; (lsu) a0  <-- 0x1001df98
            526000    0x80006040 lw      a2, 12(sp)             #; sp  = 0x1001df78, a2  <~~ Word[0x1001df84]
            527000    0x80006044 lui     a1, 0x2                #; (wrb) a1  <-- 8192
            528000    0x80006048 addi    t2, a1, 64             #; a1  = 8192, (wrb) t2  <-- 8256
            529000    0x8000604c beqz    a4, pc + 28            #; a4  = 0, taken, (lsu) a2  <-- 0x1001df98, goto 0x80006068
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:46)
#;     tls_ptr += size;
            538000    0x80006068 lw      a0, 12(sp)             #; sp  = 0x1001df78, a0  <~~ Word[0x1001df84]
            541000                                              #; (lsu) a0  <-- 0x1001df98
            542000    0x8000606c add     a0, a0, a4             #; a0  = 0x1001df98, a4  = 0, (wrb) a0  <-- 0x1001df98
            543000    0x80006070 sw      a0, 12(sp)             #; sp  = 0x1001df78, 0x1001df98 ~~> Word[0x1001df84]
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d((void*)(tls_ptr + i * tls_offset),
            544000    0x80006074 lw      a0, 12(sp)             #; sp  = 0x1001df78, a0  <~~ Word[0x1001df84]
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:13)
#;         if (size > 0) {
            545000    0x80006078 auipc   a1, 0x1                #; (wrb) a1  <-- 0x80007078
            546000    0x8000607c addi    a1, a1, -1008          #; a1  = 0x80007078, (wrb) a1  <-- 0x80006c88
            547000                                              #; (lsu) a0  <-- 0x1001df98
            550000    0x80006080 auipc   a2, 0x1                #; (wrb) a2  <-- 0x80007080
            551000    0x80006084 addi    a2, a2, -1000          #; a2  = 0x80007080, (wrb) a2  <-- 0x80006c98
            552000    0x80006088 sub     a4, a2, a1             #; a2  = 0x80006c98, a1  = 0x80006c88, (wrb) a4  <-- 16
            553000    0x8000608c bnez    a4, pc + 500           #; a4  = 16, taken, goto 0x80006280
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:21)
#;         asm volatile(
            573000    0x80006280 lui     a2, 0x10030            #; (wrb) a2  <-- 0x10030000
            574000    0x80006284 li      a3, 0                  #; (wrb) a3  <-- 0
            575000    0x80006288 dmsrc   a2, a3                 #; a2  = 0x10030000, a3  = 0
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:30)
#;         asm volatile(
            576000    0x8000628c li      a1, 0                  #; (wrb) a1  <-- 0
            577000    0x80006290 dmdst   a0, a1                 #; a0  = 0x1001df98, a1  = 0
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:40)
#;         asm volatile(
            578000    0x80006294 dmcpyi  a0, a4, 0              #; a4  = 16
            581000                                              #; (acc) a0  <-- 0
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d((void*)(tls_ptr + i * tls_offset),
            582000    0x80006298 lw      a0, 12(sp)             #; sp  = 0x1001df78, a0  <~~ Word[0x1001df84]
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:13)
#;         if (size > 0) {
            583000    0x8000629c beqz    a4, pc - 516           #; a4  = 16, not taken
            585000                                              #; (lsu) a0  <-- 0x1001df98
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d((void*)(tls_ptr + i * tls_offset),
            595000    0x800062a0 addi    a0, a0, 1032           #; a0  = 0x1001df98, (wrb) a0  <-- 0x1001e3a0
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:21)
#;         asm volatile(
            596000    0x800062a4 lui     a2, 0x10030            #; (wrb) a2  <-- 0x10030000
            597000    0x800062a8 li      a3, 0                  #; (wrb) a3  <-- 0
            598000    0x800062ac dmsrc   a2, a3                 #; a2  = 0x10030000, a3  = 0
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:30)
#;         asm volatile(
            599000    0x800062b0 li      a1, 0                  #; (wrb) a1  <-- 0
            600000    0x800062b4 dmdst   a0, a1                 #; a0  = 0x1001e3a0, a1  = 0
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:40)
#;         asm volatile(
            601000    0x800062b8 dmcpyi  a0, a4, 0              #; a4  = 16
            604000                                              #; (acc) a0  <-- 1
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d((void*)(tls_ptr + i * tls_offset),
            605000    0x800062bc lw      a0, 12(sp)             #; sp  = 0x1001df78, a0  <~~ Word[0x1001df84]
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:13)
#;         if (size > 0) {
            607000    0x800062c0 beqz    a4, pc - 544           #; a4  = 16, not taken
            608000                                              #; (lsu) a0  <-- 0x1001df98
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d((void*)(tls_ptr + i * tls_offset),
            609000    0x800062c4 add     a0, a0, a7             #; a0  = 0x1001df98, a7  = 2064, (wrb) a0  <-- 0x1001e7a8
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:21)
#;         asm volatile(
            610000    0x800062c8 lui     a2, 0x10030            #; (wrb) a2  <-- 0x10030000
            611000    0x800062cc li      a3, 0                  #; (wrb) a3  <-- 0
            612000    0x800062d0 dmsrc   a2, a3                 #; a2  = 0x10030000, a3  = 0
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:30)
#;         asm volatile(
            613000    0x800062d4 li      a1, 0                  #; (wrb) a1  <-- 0
            614000    0x800062d8 dmdst   a0, a1                 #; a0  = 0x1001e7a8, a1  = 0
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:40)
#;         asm volatile(
            615000    0x800062dc dmcpyi  a0, a4, 0              #; a4  = 16
            618000                                              #; (acc) a0  <-- 2
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d((void*)(tls_ptr + i * tls_offset),
            620000    0x800062e0 lw      a0, 12(sp)             #; sp  = 0x1001df78, a0  <~~ Word[0x1001df84]
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:13)
#;         if (size > 0) {
            621000    0x800062e4 beqz    a4, pc - 572           #; a4  = 16, not taken
            623000                                              #; (lsu) a0  <-- 0x1001df98
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d((void*)(tls_ptr + i * tls_offset),
            624000    0x800062e8 add     a0, a7, a0             #; a7  = 2064, a0  = 0x1001df98, (wrb) a0  <-- 0x1001e7a8
            625000    0x800062ec addi    a0, a0, 1032           #; a0  = 0x1001e7a8, (wrb) a0  <-- 0x1001ebb0
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:21)
#;         asm volatile(
            626000    0x800062f0 lui     a2, 0x10030            #; (wrb) a2  <-- 0x10030000
            627000    0x800062f4 li      a3, 0                  #; (wrb) a3  <-- 0
            628000    0x800062f8 dmsrc   a2, a3                 #; a2  = 0x10030000, a3  = 0
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:30)
#;         asm volatile(
            629000    0x800062fc li      a1, 0                  #; (wrb) a1  <-- 0
            634000    0x80006300 dmdst   a0, a1                 #; a0  = 0x1001ebb0, a1  = 0
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:40)
#;         asm volatile(
            635000    0x80006304 dmcpyi  a0, a4, 0              #; a4  = 16
            638000                                              #; (acc) a0  <-- 3
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d((void*)(tls_ptr + i * tls_offset),
            639000    0x80006308 lw      a0, 12(sp)             #; sp  = 0x1001df78, a0  <~~ Word[0x1001df84]
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:13)
#;         if (size > 0) {
            640000    0x8000630c beqz    a4, pc - 604           #; a4  = 16, not taken
            642000                                              #; (lsu) a0  <-- 0x1001df98
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d((void*)(tls_ptr + i * tls_offset),
            643000    0x80006310 add     a0, a0, t0             #; a0  = 0x1001df98, t0  = 4128, (wrb) a0  <-- 0x1001efb8
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:21)
#;         asm volatile(
            644000    0x80006314 lui     a2, 0x10030            #; (wrb) a2  <-- 0x10030000
            645000    0x80006318 li      a3, 0                  #; (wrb) a3  <-- 0
            646000    0x8000631c dmsrc   a2, a3                 #; a2  = 0x10030000, a3  = 0
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:30)
#;         asm volatile(
            653000    0x80006320 li      a1, 0                  #; (wrb) a1  <-- 0
            654000    0x80006324 dmdst   a0, a1                 #; a0  = 0x1001efb8, a1  = 0
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:40)
#;         asm volatile(
            655000    0x80006328 dmcpyi  a0, a4, 0              #; a4  = 16
            658000                                              #; (acc) a0  <-- 4
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d((void*)(tls_ptr + i * tls_offset),
            659000    0x8000632c lw      a0, 12(sp)             #; sp  = 0x1001df78, a0  <~~ Word[0x1001df84]
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:13)
#;         if (size > 0) {
            660000    0x80006330 beqz    a4, pc - 632           #; a4  = 16, not taken
            662000                                              #; (lsu) a0  <-- 0x1001df98
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d((void*)(tls_ptr + i * tls_offset),
            663000    0x80006334 add     a0, t0, a0             #; t0  = 4128, a0  = 0x1001df98, (wrb) a0  <-- 0x1001efb8
            664000    0x80006338 addi    a0, a0, 1032           #; a0  = 0x1001efb8, (wrb) a0  <-- 0x1001f3c0
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:21)
#;         asm volatile(
            665000    0x8000633c lui     a2, 0x10030            #; (wrb) a2  <-- 0x10030000
            673000    0x80006340 li      a3, 0                  #; (wrb) a3  <-- 0
            674000    0x80006344 dmsrc   a2, a3                 #; a2  = 0x10030000, a3  = 0
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:30)
#;         asm volatile(
            675000    0x80006348 li      a1, 0                  #; (wrb) a1  <-- 0
            676000    0x8000634c dmdst   a0, a1                 #; a0  = 0x1001f3c0, a1  = 0
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:40)
#;         asm volatile(
            677000    0x80006350 dmcpyi  a0, a4, 0              #; a4  = 16
            680000                                              #; (acc) a0  <-- 5
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d((void*)(tls_ptr + i * tls_offset),
            681000    0x80006354 lw      a0, 12(sp)             #; sp  = 0x1001df78, a0  <~~ Word[0x1001df84]
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:13)
#;         if (size > 0) {
            682000    0x80006358 beqz    a4, pc - 664           #; a4  = 16, not taken
            684000                                              #; (lsu) a0  <-- 0x1001df98
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d((void*)(tls_ptr + i * tls_offset),
            685000    0x8000635c add     a0, a0, t1             #; a0  = 0x1001df98, t1  = 6192, (wrb) a0  <-- 0x1001f7c8
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:21)
#;         asm volatile(
            697000    0x80006360 lui     a2, 0x10030            #; (wrb) a2  <-- 0x10030000
            698000    0x80006364 li      a3, 0                  #; (wrb) a3  <-- 0
            699000    0x80006368 dmsrc   a2, a3                 #; a2  = 0x10030000, a3  = 0
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:30)
#;         asm volatile(
            700000    0x8000636c li      a1, 0                  #; (wrb) a1  <-- 0
            701000    0x80006370 dmdst   a0, a1                 #; a0  = 0x1001f7c8, a1  = 0
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:40)
#;         asm volatile(
            702000    0x80006374 dmcpyi  a0, a4, 0              #; a4  = 16
            705000                                              #; (acc) a0  <-- 6
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d((void*)(tls_ptr + i * tls_offset),
            706000    0x80006378 lw      a0, 12(sp)             #; sp  = 0x1001df78, a0  <~~ Word[0x1001df84]
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:13)
#;         if (size > 0) {
            707000    0x8000637c beqz    a4, pc - 692           #; a4  = 16, not taken
            709000                                              #; (lsu) a0  <-- 0x1001df98
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d((void*)(tls_ptr + i * tls_offset),
            719000    0x80006380 add     a0, t1, a0             #; t1  = 6192, a0  = 0x1001df98, (wrb) a0  <-- 0x1001f7c8
            720000    0x80006384 addi    a0, a0, 1032           #; a0  = 0x1001f7c8, (wrb) a0  <-- 0x1001fbd0
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:21)
#;         asm volatile(
            721000    0x80006388 lui     a2, 0x10030            #; (wrb) a2  <-- 0x10030000
            722000    0x8000638c li      a3, 0                  #; (wrb) a3  <-- 0
            723000    0x80006390 dmsrc   a2, a3                 #; a2  = 0x10030000, a3  = 0
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:30)
#;         asm volatile(
            724000    0x80006394 li      a1, 0                  #; (wrb) a1  <-- 0
            725000    0x80006398 dmdst   a0, a1                 #; a0  = 0x1001fbd0, a1  = 0
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:40)
#;         asm volatile(
            726000    0x8000639c dmcpyi  a0, a4, 0              #; a4  = 16
            729000                                              #; (acc) a0  <-- 7
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d((void*)(tls_ptr + i * tls_offset),
            731000    0x800063a0 lw      a0, 12(sp)             #; sp  = 0x1001df78, a0  <~~ Word[0x1001df84]
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:13)
#;         if (size > 0) {
            732000    0x800063a4 bnez    a4, pc - 724           #; a4  = 16, taken, goto 0x800060d0
            734000                                              #; (lsu) a0  <-- 0x1001df98
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d((void*)(tls_ptr + i * tls_offset),
            735000    0x800060d0 add     a0, a0, t2             #; a0  = 0x1001df98, t2  = 8256, (wrb) a0  <-- 0x1001ffd8
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:21)
#;         asm volatile(
            736000    0x800060d4 lui     a2, 0x10030            #; (wrb) a2  <-- 0x10030000
            737000    0x800060d8 li      a3, 0                  #; (wrb) a3  <-- 0
            738000    0x800060dc dmsrc   a2, a3                 #; a2  = 0x10030000, a3  = 0
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:30)
#;         asm volatile(
            739000    0x800060e0 li      a1, 0                  #; (wrb) a1  <-- 0
            740000    0x800060e4 dmdst   a0, a1                 #; a0  = 0x1001ffd8, a1  = 0
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:49)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:40)
#;         asm volatile(
            741000    0x800060e8 dmcpyi  a0, a4, 0              #; a4  = 16
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:54)
#;     snrt_cluster_hw_barrier (sync.h:59)
#;       asm volatile("csrr x0, 0x7C2" ::: "memory");
            742000    0x800060ec csrs    unknown_7c2, zero      #; csr@7c2 = 0
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:-1)
#; 
            744000    0x800060f0 sltiu   a7, a5, 8              #; a5  = 8, (wrb) a7  <-- 0
#; snrt_main (start.c:122)
#;   snrt_init_bss (start.c:63)
#;     if (snrt_cluster_idx() == 0 && snrt_is_dm_core()) {
            745000    0x800060f4 bltu    a6, s2, pc + 96        #; a6  = 8, s2  = 8, not taken, (acc) a0  <-- 8
#; snrt_main (start.c:122)
#;   snrt_init_bss (start.c:63)
#;     snrt_is_dm_core (team.h:57)
#;       snrt_is_compute_core (team.h:53)
#;         snrt_cluster_core_idx (team.h:41)
#;           return snrt_global_core_idx() % snrt_cluster_core_num();
            746000    0x800060f8 andi    a0, s2, 255            #; s2  = 8, (wrb) a0  <-- 8
            747000    0x800060fc lui     a1, 0x38e39            #; (wrb) a1  <-- 0x38e39000
            748000    0x80006100 addi    a1, a1, -455           #; a1  = 0x38e39000, (wrb) a1  <-- 0x38e38e39
            749000    0x80006104 mulhu   a1, a0, a1             #; a0  = 8, a1  = 0x38e38e39
            753000                                              #; (acc) a1  <-- 1
            754000    0x80006108 srli    a1, a1, 1              #; a1  = 1, (wrb) a1  <-- 0
            755000    0x8000610c slli    a2, a1, 3              #; a1  = 0, (wrb) a2  <-- 0
            756000    0x80006110 add     a1, a2, a1             #; a2  = 0, a1  = 0, (wrb) a1  <-- 0
            757000    0x80006114 sub     a0, a0, a1             #; a0  = 8, a1  = 0, (wrb) a0  <-- 8
#; snrt_main (start.c:122)
#;   snrt_init_bss (start.c:63)
#;     snrt_is_dm_core (team.h:57)
#;       snrt_is_compute_core (team.h:53)
#;         return snrt_cluster_core_idx() < snrt_cluster_compute_core_num();
            758000    0x80006118 sltiu   a1, a0, 8              #; a0  = 8, (wrb) a1  <-- 0
#; snrt_main (start.c:122)
#;   snrt_init_bss (start.c:63)
#;     if (snrt_cluster_idx() == 0 && snrt_is_dm_core()) {
            759000    0x8000611c auipc   a0, 0x1                #; (wrb) a0  <-- 0x8000711c
            760000    0x80006120 addi    a0, a0, -292           #; a0  = 0x8000711c, (wrb) a0  <-- 0x80006ff8
            761000    0x80006124 auipc   a2, 0x1                #; (wrb) a2  <-- 0x80007124
            762000    0x80006128 addi    a2, a2, 512            #; a2  = 0x80007124, (wrb) a2  <-- 0x80007324
            763000    0x8000612c sub     a4, a2, a0             #; a2  = 0x80007324, a0  = 0x80006ff8, (wrb) a4  <-- 812
            764000    0x80006130 seqz    a2, a4                 #; a4  = 812, (wrb) a2  <-- 0
            765000    0x80006134 or      a1, a1, a2             #; a1  = 0, a2  = 0, (wrb) a1  <-- 0
            766000    0x80006138 bnez    a1, pc + 28            #; a1  = 0, not taken
#; snrt_main (start.c:122)
#;   snrt_init_bss (start.c:65)
#;     snrt_dma_start_1d_wideptr (dma.h:21)
#;       asm volatile(
            767000    0x8000613c lui     a2, 0x10030            #; (wrb) a2  <-- 0x10030000
            772000    0x80006140 li      a3, 0                  #; (wrb) a3  <-- 0
            773000    0x80006144 dmsrc   a2, a3                 #; a2  = 0x10030000, a3  = 0
#; snrt_main (start.c:122)
#;   snrt_init_bss (start.c:65)
#;     snrt_dma_start_1d_wideptr (dma.h:30)
#;       asm volatile(
            774000    0x80006148 li      a1, 0                  #; (wrb) a1  <-- 0
            775000    0x8000614c dmdst   a0, a1                 #; a0  = 0x80006ff8, a1  = 0
#; snrt_main (start.c:122)
#;   snrt_init_bss (start.c:65)
#;     snrt_dma_start_1d_wideptr (dma.h:40)
#;       asm volatile(
            776000    0x80006150 dmcpyi  a0, a4, 0              #; a4  = 812
#; snrt_main (start.c:130)
#;   snrt_init_cls (start.c:76)
#;     _cls_ptr = (cls_t*)snrt_cls_base_addr();
            777000    0x80006154 auipc   a5, 0x1                #; (wrb) a5  <-- 0x80007154
            778000    0x80006158 addi    a5, a5, -1212          #; a5  = 0x80007154, (wrb) a5  <-- 0x80006c98
            779000    0x8000615c auipc   t0, 0x1                #; (wrb) t0  <-- 0x8000715c
            780000                                              #; (acc) a0  <-- 9
            784000    0x80006160 addi    t0, t0, -1220          #; t0  = 0x8000715c, (wrb) t0  <-- 0x80006c98
            785000    0x80006164 sub     a4, t0, a5             #; t0  = 0x80006c98, a5  = 0x80006c98, (wrb) a4  <-- 0
            786000    0x80006168 auipc   s0, 0x1                #; (wrb) s0  <-- 0x80007168
            787000    0x8000616c addi    s0, s0, -1216          #; s0  = 0x80007168, (wrb) s0  <-- 0x80006ca8
            788000    0x80006170 add     a0, a4, s0             #; a4  = 0, s0  = 0x80006ca8, (wrb) a0  <-- 0x80006ca8
#; snrt_main (start.c:130)
#;   snrt_init_cls (start.c:-1)
#; 
            789000    0x80006174 auipc   s1, 0x1                #; (wrb) s1  <-- 0x80007174
            790000    0x80006178 addi    s1, s1, -1244          #; s1  = 0x80007174, (wrb) s1  <-- 0x80006c98
#; snrt_main (start.c:130)
#;   snrt_init_cls (start.c:76)
#;     _cls_ptr = (cls_t*)snrt_cls_base_addr();
            791000    0x8000617c sub     a0, s1, a0             #; s1  = 0x80006c98, a0  = 0x80006ca8, (wrb) a0  <-- -16
            796000    0x80006180 lui     a1, 0x10020            #; (wrb) a1  <-- 0x10020000
            797000    0x80006184 add     a1, a0, a1             #; a0  = -16, a1  = 0x10020000, (wrb) a1  <-- 0x1001fff0
            798000    0x80006188 lui     a2, 0x0                #; (wrb) a2  <-- 0
            799000    0x8000618c add     a2, a2, tp             #; a2  = 0, tp  = 0x1001df98, (wrb) a2  <-- 0x1001df98
            800000    0x80006190 sw      a1, 0(a2)              #; a2  = 0x1001df98, 0x1001fff0 ~~> Word[0x1001df98]
#; snrt_main (start.c:130)
#;   snrt_init_cls (start.c:79)
#;     if (snrt_is_dm_core()) {
            801000    0x80006194 bnez    a7, pc + 172           #; a7  = 0, not taken
#; snrt_main (start.c:130)
#;   snrt_init_cls (start.c:85)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:13)
#;         if (size > 0) {
            802000    0x80006198 beqz    a4, pc + 36            #; a4  = 0, taken, goto 0x800061bc
#; snrt_main (start.c:130)
#;   snrt_init_cls (start.c:90)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:13)
#;         if (size > 0) {
            809000    0x800061bc sub     a6, s0, s1             #; s0  = 0x80006ca8, s1  = 0x80006c98, (wrb) a6  <-- 16
            824000    0x800061c0 beqz    a6, pc + 52            #; a6  = 16, not taken
#; snrt_main (start.c:130)
#;   snrt_init_cls (start.c:90)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:21)
#;         asm volatile(
            825000    0x800061c4 lui     a2, 0x10030            #; (wrb) a2  <-- 0x10030000
#; snrt_main (start.c:130)
#;   snrt_init_cls (start.c:90)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:30)
#;         asm volatile(
            826000    0x800061c8 add     a0, a4, a6             #; a4  = 0, a6  = 16, (wrb) a0  <-- 16
            827000    0x800061cc add     a0, a0, a5             #; a0  = 16, a5  = 0x80006c98, (wrb) a0  <-- 0x80006ca8
            828000    0x800061d0 sub     a0, t0, a0             #; t0  = 0x80006c98, a0  = 0x80006ca8, (wrb) a0  <-- -16
            829000    0x800061d4 lui     a1, 0x10020            #; (wrb) a1  <-- 0x10020000
            830000    0x800061d8 add     a0, a0, a1             #; a0  = -16, a1  = 0x10020000, (wrb) a0  <-- 0x1001fff0
#; snrt_main (start.c:130)
#;   snrt_init_cls (start.c:90)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:21)
#;         asm volatile(
            831000    0x800061dc li      a3, 0                  #; (wrb) a3  <-- 0
            839000    0x800061e0 dmsrc   a2, a3                 #; a2  = 0x10030000, a3  = 0
#; snrt_main (start.c:130)
#;   snrt_init_cls (start.c:90)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:30)
#;         asm volatile(
            840000    0x800061e4 li      a1, 0                  #; (wrb) a1  <-- 0
            841000    0x800061e8 dmdst   a0, a1                 #; a0  = 0x1001fff0, a1  = 0
#; snrt_main (start.c:130)
#;   snrt_init_cls (start.c:90)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:40)
#;         asm volatile(
            842000    0x800061ec mv      a4, a6                 #; a6  = 16, (wrb) a4  <-- 16
            843000    0x800061f0 dmcpyi  a0, a4, 0              #; a4  = 16
#; snrt_main (start.c:135)
#;   snrt_dma_wait_all (dma.h:160)
#;     asm volatile(
            844000    0x800061f4 dmstati t0, 2                  #; 
            846000                                              #; (acc) a0  <-- 10
            847000                                              #; (acc) t0  <-- 1
            848000    0x800061f8 bnez    t0, pc - 4             #; t0  = 1, taken, goto 0x800061f4
            849000    0x800061f4 dmstati t0, 2                  #; 
            852000                                              #; (acc) t0  <-- 1
            853000    0x800061f8 bnez    t0, pc - 4             #; t0  = 1, taken, goto 0x800061f4
            854000    0x800061f4 dmstati t0, 2                  #; 
            857000                                              #; (acc) t0  <-- 1
            858000    0x800061f8 bnez    t0, pc - 4             #; t0  = 1, taken, goto 0x800061f4
            859000    0x800061f4 dmstati t0, 2                  #; 
            862000                                              #; (acc) t0  <-- 0
            863000    0x800061f8 bnez    t0, pc - 4             #; t0  = 0, not taken
#; snrt_main (start.c:143)
#;   snrt_init_libs (start.c:96)
#;     snrt_alloc_init (alloc.h:82)
#;       snrt_l1_allocator (alloc.h:13)
#;         cls (cls.h:7)
#;           inline cls_t* cls() { return _cls_ptr; }
            864000    0x800061fc lui     a0, 0x0                #; (wrb) a0  <-- 0
            876000    0x80006200 add     a0, a0, tp             #; a0  = 0, tp  = 0x1001df98, (wrb) a0  <-- 0x1001df98
            877000    0x80006204 lw      a0, 0(a0)              #; a0  = 0x1001df98, a0  <~~ Word[0x1001df98]
            878000    0x80006208 lui     a1, 0x10000            #; (wrb) a1  <-- 0x10000000
            880000                                              #; (lsu) a0  <-- 0x1001fff0
#; snrt_main (start.c:143)
#;   snrt_init_libs (start.c:96)
#;     snrt_alloc_init (alloc.h:82)
#;       snrt_l1_allocator()->base =
            881000    0x8000620c sw      a1, 4(a0)              #; a0  = 0x1001fff0, 0x10000000 ~~> Word[0x1001fff4]
            882000    0x80006210 lui     a2, 0x20               #; (wrb) a2  <-- 0x00020000
#; snrt_main (start.c:143)
#;   snrt_init_libs (start.c:96)
#;     snrt_alloc_init (alloc.h:84)
#;       snrt_l1_allocator()->size = snrt_l1_end_addr() - snrt_l1_start_addr();
            883000    0x80006214 sw      a2, 8(a0)              #; a0  = 0x1001fff0, 0x00020000 ~~> Word[0x1001fff8]
#; snrt_main (start.c:143)
#;   snrt_init_libs (start.c:96)
#;     snrt_alloc_init (alloc.h:85)
#;       snrt_l1_allocator()->next = snrt_l1_allocator()->base;
            884000    0x80006218 sw      a1, 12(a0)             #; a0  = 0x1001fff0, 0x10000000 ~~> Word[0x1001fffc]
#; snrt_main (start.c:143)
#;   snrt_init_libs (start.c:96)
#;     snrt_alloc_init (alloc.h:-1)
#; 
            885000    0x8000621c auipc   a0, 0x1                #; (wrb) a0  <-- 0x8000721c
            888000    0x80006220 addi    a0, a0, 264            #; a0  = 0x8000721c, (wrb) a0  <-- 0x80007324
            889000    0x80006224 addi    a0, a0, 7              #; a0  = 0x80007324, (wrb) a0  <-- 0x8000732b
#; snrt_main (start.c:143)
#;   snrt_init_libs (start.c:96)
#;     snrt_alloc_init (alloc.h:88)
#;       snrt_l3_allocator()->base = ALIGN_UP((uint32_t)&_edram, MIN_CHUNK_SIZE);
            890000    0x80006228 andi    a0, a0, -8             #; a0  = 0x8000732b, (wrb) a0  <-- 0x80007328
#; snrt_main (start.c:143)
#;   snrt_init_libs (start.c:96)
#;     snrt_alloc_init (alloc.h:-1)
#; 
            891000    0x8000622c auipc   a1, 0x1                #; (wrb) a1  <-- 0x8000722c
            892000    0x80006230 addi    a1, a1, 236            #; a1  = 0x8000722c, (wrb) a1  <-- 0x80007318
#; snrt_main (start.c:143)
#;   snrt_init_libs (start.c:96)
#;     snrt_alloc_init (alloc.h:88)
#;       snrt_l3_allocator()->base = ALIGN_UP((uint32_t)&_edram, MIN_CHUNK_SIZE);
            893000    0x80006234 sw      a0, 0(a1)              #; a1  = 0x80007318, 0x80007328 ~~> Word[0x80007318]
#; snrt_main (start.c:143)
#;   snrt_init_libs (start.c:96)
#;     snrt_alloc_init (alloc.h:89)
#;       snrt_l3_allocator()->size = 0;
            894000    0x80006238 sw      zero, 4(a1)            #; a1  = 0x80007318, 0 ~~> Word[0x8000731c]
#; snrt_main (start.c:143)
#;   snrt_init_libs (start.c:96)
#;     snrt_alloc_init (alloc.h:90)
#;       snrt_l3_allocator()->next = snrt_l3_allocator()->base;
            895000    0x8000623c sw      a0, 8(a1)              #; a1  = 0x80007318, 0x80007328 ~~> Word[0x80007320]
#; snrt_main (start.c:151)
#;   snrt_cluster_hw_barrier (sync.h:59)
#;     asm volatile("csrr x0, 0x7C2" ::: "memory");
            896000    0x80006240 csrs    unknown_7c2, zero      #; csr@7c2 = 0
#; snrt_main (start.c:160)
#;   exit_code = main();
            898000    0x80006244 auipc   ra, 0xffffc            #; (wrb) ra  <-- 0x80002244
            899000    0x80006248 jalr    ra, ra, -448           #; ra  = 0x80002244, (wrb) ra  <-- 0x8000624c, goto 0x80002084
#; main (main.c:9)
#;   int main() {
            911000    0x80002084 addi    sp, sp, -48            #; sp  = 0x1001df78, (wrb) sp  <-- 0x1001df48
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:-1)
#; 
            912000    0x80002088 sw      s0, 44(sp)             #; sp  = 0x1001df48, 0x80006ca8 ~~> Word[0x1001df74]
            913000    0x8000208c sw      s1, 40(sp)             #; sp  = 0x1001df48, 0x80006c98 ~~> Word[0x1001df70]
            914000    0x80002090 sw      s2, 36(sp)             #; sp  = 0x1001df48, 8 ~~> Word[0x1001df6c]
            917000    0x80002094 sw      s3, 32(sp)             #; sp  = 0x1001df48, 0 ~~> Word[0x1001df68]
            918000    0x80002098 sw      s4, 28(sp)             #; sp  = 0x1001df48, 0 ~~> Word[0x1001df64]
            920000    0x8000209c sw      s5, 24(sp)             #; sp  = 0x1001df48, 0 ~~> Word[0x1001df60]
            923000    0x800020a0 sw      s6, 20(sp)             #; sp  = 0x1001df48, 0 ~~> Word[0x1001df5c]
            924000    0x800020a4 sw      s7, 16(sp)             #; sp  = 0x1001df48, 0 ~~> Word[0x1001df58]
            925000    0x800020a8 sw      s8, 12(sp)             #; sp  = 0x1001df48, 0 ~~> Word[0x1001df54]
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:815)
#;     snrt_cluster_core_idx (team.h:41)
#;       snrt_global_core_idx (team.h:28)
#;         snrt_hartid (team.h:7)
#;           asm("csrr %0, mhartid" : "=r"(hartid));
            926000    0x800020ac csrr    t2, mhartid            #; mhartid = 8, (wrb) t2  <-- 8
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:815)
#;     snrt_cluster_core_idx (team.h:41)
#;       snrt_global_core_idx (team.h:28)
#;         snrt_hartid (team.h:-1)
#; 
            927000    0x800020b0 lui     a0, 0x38e39            #; (wrb) a0  <-- 0x38e39000
            928000    0x800020b4 addi    a0, a0, -455           #; a0  = 0x38e39000, (wrb) a0  <-- 0x38e38e39
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:815)
#;     snrt_cluster_core_idx (team.h:41)
#;       return snrt_global_core_idx() % snrt_cluster_core_num();
            929000    0x800020b8 mulhu   a0, t2, a0             #; t2  = 8, a0  = 0x38e38e39
            934000                                              #; (acc) a0  <-- 1
            935000    0x800020bc srli    a0, a0, 1              #; a0  = 1, (wrb) a0  <-- 0
            936000    0x800020c0 slli    a1, a0, 3              #; a0  = 0, (wrb) a1  <-- 0
            937000    0x800020c4 add     a0, a1, a0             #; a1  = 0, a0  = 0, (wrb) a0  <-- 0
#; main (main.c:10)
#;   batchnorm_backward_training (team.h:-1)
#; 
            938000    0x800020c8 auipc   t3, 0x5                #; (wrb) t3  <-- 0x800070c8
            939000    0x800020cc addi    t3, t3, -360           #; t3  = 0x800070c8, (wrb) t3  <-- 0x80006f60
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:819)
#;     uint32_t H = l->IH;
            940000    0x800020d0 lw      a7, 4(t3)              #; t3  = 0x80006f60, a7  <~~ Word[0x80006f64]
            961000                                              #; (lsu) a7  <-- 2
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:820)
#;     uint32_t W = l->IW;
            962000    0x800020d4 lw      t0, 8(t3)              #; t3  = 0x80006f60, t0  <~~ Word[0x80006f68]
           1010000                                              #; (lsu) t0  <-- 2
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:821)
#;     uint32_t C = l->CI;
           1011000    0x800020d8 lw      s2, 0(t3)              #; t3  = 0x80006f60, s2  <~~ Word[0x80006f60]
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:875)
#;     snrt_mcycle (riscv.h:17)
#;       asm volatile("csrr %0, mcycle" : "=r"(r) : : "memory");
           1012000    0x800020dc csrr    a1, mcycle             #; mcycle = 1008, (wrb) a1  <-- 1008
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:815)
#;     snrt_cluster_core_idx (team.h:41)
#;       return snrt_global_core_idx() % snrt_cluster_core_num();
           1013000    0x800020e0 sub     s4, t2, a0             #; t2  = 8, a0  = 0, (wrb) s4  <-- 8
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:815)
#;     snrt_cluster_core_idx (team.h:-1)
#; 
           1014000    0x800020e4 li      a0, 8                  #; (wrb) a0  <-- 8
           1051000                                              #; (lsu) s2  <-- 4
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:835)
#;     ptr += C;
           1052000    0x800020e8 slli    a4, s2, 3              #; s2  = 4, (wrb) a4  <-- 32
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:876)
#;     if (snrt_is_dm_core()) {
           1053000    0x800020ec bgeu    s4, a0, pc + 340       #; s4  = 8, a0  = 8, taken, goto 0x80002240
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:877)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:13)
#;         if (size > 0) {
           1065000    0x80002240 beqz    a4, pc + 36            #; a4  = 32, not taken
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:877)
#;     curr_var_load = snrt_dma_start_1d(invstd, l->current_var,
           1066000    0x80002244 lw      a2, 24(t3)             #; t3  = 0x80006f60, a2  <~~ Word[0x80006f78]
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:877)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:21)
#;         asm volatile(
           1067000    0x80002248 li      a3, 0                  #; (wrb) a3  <-- 0
           1081000                                              #; (lsu) a2  <-- 0x80006fb8
           1082000    0x8000224c dmsrc   a2, a3                 #; a2  = 0x80006fb8, a3  = 0
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:877)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:30)
#;         asm volatile(
           1083000    0x80002250 lui     a0, 0x10000            #; (wrb) a0  <-- 0x10000000
           1084000    0x80002254 li      a1, 0                  #; (wrb) a1  <-- 0
           1085000    0x80002258 dmdst   a0, a1                 #; a0  = 0x10000000, a1  = 0
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:877)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:40)
#;         asm volatile(
           1086000    0x8000225c dmcpyi  a0, a4, 0              #; a4  = 32
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:877)
#;     snrt_dma_start_1d (dma.h:59)
#;       snrt_dma_start_1d_wideptr (dma.h:-1)
#; 
           1087000    0x80002260 j       pc + 0x8               #; goto 0x80002268
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:879)
#;     snrt_dma_wait (dma.h:145)
#;       asm volatile(
           1088000    0x80002268 dmstati t0, 0                  #; 
           1089000                                              #; (acc) a0  <-- 11
           1091000                                              #; (acc) t0  <-- 11
           1092000    0x8000226c sub     t0, t0, a0             #; t0  = 11, a0  = 11, (wrb) t0  <-- 0
           1093000    0x80002270 bge     zero, t0, pc - 8       #; t0  = 0, taken, goto 0x80002268
           1094000    0x80002268 dmstati t0, 0                  #; 
           1097000                                              #; (acc) t0  <-- 11
           1098000    0x8000226c sub     t0, t0, a0             #; t0  = 11, a0  = 11, (wrb) t0  <-- 0
           1099000    0x80002270 bge     zero, t0, pc - 8       #; t0  = 0, taken, goto 0x80002268
           1100000    0x80002268 dmstati t0, 0                  #; 
           1103000                                              #; (acc) t0  <-- 11
           1104000    0x8000226c sub     t0, t0, a0             #; t0  = 11, a0  = 11, (wrb) t0  <-- 0
           1105000    0x80002270 bge     zero, t0, pc - 8       #; t0  = 0, taken, goto 0x80002268
           1106000    0x80002268 dmstati t0, 0                  #; 
           1109000                                              #; (acc) t0  <-- 11
           1110000    0x8000226c sub     t0, t0, a0             #; t0  = 11, a0  = 11, (wrb) t0  <-- 0
           1111000    0x80002270 bge     zero, t0, pc - 8       #; t0  = 0, taken, goto 0x80002268
           1112000    0x80002268 dmstati t0, 0                  #; 
           1115000                                              #; (acc) t0  <-- 12
           1116000    0x8000226c sub     t0, t0, a0             #; t0  = 12, a0  = 11, (wrb) t0  <-- 1
           1117000    0x80002270 bge     zero, t0, pc - 8       #; t0  = 1, not taken
#; main (dma.h:-1)
#; 
           1118000    0x80002274 auipc   a0, 0x5                #; (wrb) a0  <-- 0x80007274
           1119000    0x80002278 addi    a0, a0, -616           #; a0  = 0x80007274, (wrb) a0  <-- 0x8000700c
#; main (main.c:12)
#;   snrt_global_barrier (sync.h:67)
#;     uint32_t prev_barrier_iteration = _snrt_barrier.iteration;
           1120000    0x8000227c lw      a1, 4(a0)              #; a0  = 0x8000700c, a1  <~~ Word[0x80007010]
#; main (main.c:12)
#;   snrt_global_barrier (sync.h:-1)
#; 
           1130000    0x80002280 li      a2, 1                  #; (wrb) a2  <-- 1
           1131000                                              #; (lsu) a1  <-- 0
#; main (main.c:12)
#;   snrt_global_barrier (sync.h:69)
#;     __atomic_add_fetch(&(_snrt_barrier.cnt), 1, __ATOMIC_RELAXED);
           1132000    0x80002284 amoadd.w a3, a2, (a0)          #; a0  = 0x8000700c, a2  = 1, a3  <~~ Word[0x8000700c]
           1150000                                              #; (lsu) a3  <-- 0
#; main (main.c:12)
#;   snrt_global_barrier (sync.h:72)
#;     if (cnt == snrt_cluster_num()) {
           1151000    0x80002288 beqz    a3, pc + 16            #; a3  = 0, taken, goto 0x80002298
#; main (main.c:12)
#;   snrt_global_barrier (sync.h:73)
#;     _snrt_barrier.cnt = 0;
           1152000    0x80002298 sw      zero, 0(a0)            #; a0  = 0x8000700c, 0 ~~> Word[0x8000700c]
           1153000    0x8000229c addi    a0, a0, 4              #; a0  = 0x8000700c, (wrb) a0  <-- 0x80007010
#; main (main.c:12)
#;   snrt_global_barrier (sync.h:74)
#;     __atomic_add_fetch(&(_snrt_barrier.iteration), 1, __ATOMIC_RELAXED);
           1164000    0x800022a0 amoadd.w a0, a2, (a0)          #; a0  = 0x80007010, a2  = 1, a0  <~~ Word[0x80007010]
           1165000    0x800022a4 j       pc + 0x328             #; goto 0x800025cc
#; main (main.c:12)
#;   snrt_global_barrier (sync.h:81)
#;     snrt_cluster_hw_barrier (sync.h:59)
#;       asm volatile("csrr x0, 0x7C2" ::: "memory");
           1176000    0x800025cc csrs    unknown_7c2, zero      #; csr@7c2 = 0
           1182000                                              #; (lsu) a0  <-- 0
#; main (main.c:14)
#;   return 0;
           1183000    0x800025d0 li      a0, 0                  #; (wrb) a0  <-- 0
           1184000    0x800025d4 lw      s8, 12(sp)             #; sp  = 0x1001df48, s8  <~~ Word[0x1001df54]
           1187000                                              #; (lsu) s8  <-- 0
           1188000    0x800025d8 lw      s7, 16(sp)             #; sp  = 0x1001df48, s7  <~~ Word[0x1001df58]
           1191000                                              #; (lsu) s7  <-- 0
           1192000    0x800025dc lw      s6, 20(sp)             #; sp  = 0x1001df48, s6  <~~ Word[0x1001df5c]
           1195000                                              #; (lsu) s6  <-- 0
           1196000    0x800025e0 lw      s5, 24(sp)             #; sp  = 0x1001df48, s5  <~~ Word[0x1001df60]
           1199000                                              #; (lsu) s5  <-- 0
           1200000    0x800025e4 lw      s4, 28(sp)             #; sp  = 0x1001df48, s4  <~~ Word[0x1001df64]
           1203000                                              #; (lsu) s4  <-- 0
           1204000    0x800025e8 lw      s3, 32(sp)             #; sp  = 0x1001df48, s3  <~~ Word[0x1001df68]
           1207000                                              #; (lsu) s3  <-- 0
           1208000    0x800025ec lw      s2, 36(sp)             #; sp  = 0x1001df48, s2  <~~ Word[0x1001df6c]
           1211000                                              #; (lsu) s2  <-- 8
           1212000    0x800025f0 lw      s1, 40(sp)             #; sp  = 0x1001df48, s1  <~~ Word[0x1001df70]
           1215000                                              #; (lsu) s1  <-- 0x80006c98
           1216000    0x800025f4 lw      s0, 44(sp)             #; sp  = 0x1001df48, s0  <~~ Word[0x1001df74]
           1217000    0x800025f8 addi    sp, sp, 48             #; sp  = 0x1001df48, (wrb) sp  <-- 0x1001df78
           1218000    0x800025fc ret                            #; ra  = 0x8000624c, goto 0x8000624c
           1219000                                              #; (lsu) s0  <-- 0x80006ca8
#; snrt_main (start.c:168)
#;   snrt_cluster_hw_barrier (sync.h:59)
#;     asm volatile("csrr x0, 0x7C2" ::: "memory");
           1231000    0x8000624c csrs    unknown_7c2, zero      #; csr@7c2 = 0
#; snrt_main (start.c:176)
#;   snrt_exit (start.c:101)
#;     if (snrt_global_core_idx() == 0)
           1285000    0x80006250 bnez    s2, pc + 24            #; s2  = 8, taken, goto 0x80006268
#; snrt_main (start.c:182)
#;   }
           1286000    0x80006268 lw      s2, 16(sp)             #; sp  = 0x1001df78, s2  <~~ Word[0x1001df88]
           1289000                                              #; (lsu) s2  <-- 0
           1290000    0x8000626c lw      s1, 20(sp)             #; sp  = 0x1001df78, s1  <~~ Word[0x1001df8c]
           1293000                                              #; (lsu) s1  <-- 0
