#; unknown function (??:0)
#; 
             33000    0x00001000 csrr    a0, mhartid            #; mhartid = 7, (wrb) a0  <-- 7
             57000    0x00001004 auipc   a1, 0x0                #; (wrb) a1  <-- 4100
             81000    0x00001008 addi    a1, a1, 32             #; a1  = 4100, (wrb) a1  <-- 4132
            105000    0x0000100c auipc   t0, 0x0                #; (wrb) t0  <-- 4108
            129000    0x00001010 lw      t0, 20(t0)             #; t0  = 4108, t0  <~~ Word[0x00001020]
            154000                                              #; (lsu) t0  <-- 0x80000000
            156000    0x00001014 jalr    t0                     #; t0  = 0x80000000, (wrb) ra  <-- 4120, goto 0x80000000
#; unknown function (start.S:12)
#;   mv t0, x0
            161000    0x80000000 li      t0, 0                  #; (wrb) t0  <-- 0
#; unknown function (start.S:13)
#;   mv t1, x0
            162000    0x80000004 li      t1, 0                  #; (wrb) t1  <-- 0
#; unknown function (start.S:14)
#;   mv t2, x0
            163000    0x80000008 li      t2, 0                  #; (wrb) t2  <-- 0
#; unknown function (start.S:15)
#;   mv t3, x0
            164000    0x8000000c li      t3, 0                  #; (wrb) t3  <-- 0
#; unknown function (start.S:16)
#;   mv t4, x0
            165000    0x80000010 li      t4, 0                  #; (wrb) t4  <-- 0
#; unknown function (start.S:17)
#;   mv t5, x0
            166000    0x80000014 li      t5, 0                  #; (wrb) t5  <-- 0
#; unknown function (start.S:18)
#;   mv t6, x0
            167000    0x80000018 li      t6, 0                  #; (wrb) t6  <-- 0
#; unknown function (start.S:19)
#;   mv a0, x0
            168000    0x8000001c li      a0, 0                  #; (wrb) a0  <-- 0
#; unknown function (start.S:20)
#;   mv a1, x0
            169000    0x80000020 li      a1, 0                  #; (wrb) a1  <-- 0
#; unknown function (start.S:21)
#;   mv a2, x0
            170000    0x80000024 li      a2, 0                  #; (wrb) a2  <-- 0
#; unknown function (start.S:22)
#;   mv a3, x0
            171000    0x80000028 li      a3, 0                  #; (wrb) a3  <-- 0
#; unknown function (start.S:23)
#;   mv a4, x0
            172000    0x8000002c li      a4, 0                  #; (wrb) a4  <-- 0
#; unknown function (start.S:24)
#;   mv a5, x0
            173000    0x80000030 li      a5, 0                  #; (wrb) a5  <-- 0
#; unknown function (start.S:25)
#;   mv a6, x0
            174000    0x80000034 li      a6, 0                  #; (wrb) a6  <-- 0
#; unknown function (start.S:26)
#;   mv a7, x0
            175000    0x80000038 li      a7, 0                  #; (wrb) a7  <-- 0
#; unknown function (start.S:27)
#;   mv s0, x0
            176000    0x8000003c li      s0, 0                  #; (wrb) s0  <-- 0
#; unknown function (start.S:28)
#;   mv s1, x0
            177000    0x80000040 li      s1, 0                  #; (wrb) s1  <-- 0
#; unknown function (start.S:29)
#;   mv s2, x0
            178000    0x80000044 li      s2, 0                  #; (wrb) s2  <-- 0
#; unknown function (start.S:30)
#;   mv s3, x0
            179000    0x80000048 li      s3, 0                  #; (wrb) s3  <-- 0
#; unknown function (start.S:31)
#;   mv s4, x0
            180000    0x8000004c li      s4, 0                  #; (wrb) s4  <-- 0
#; unknown function (start.S:32)
#;   mv s5, x0
            181000    0x80000050 li      s5, 0                  #; (wrb) s5  <-- 0
#; unknown function (start.S:33)
#;   mv s6, x0
            182000    0x80000054 li      s6, 0                  #; (wrb) s6  <-- 0
#; unknown function (start.S:34)
#;   mv s7, x0
            183000    0x80000058 li      s7, 0                  #; (wrb) s7  <-- 0
#; unknown function (start.S:35)
#;   mv s8, x0
            184000    0x8000005c li      s8, 0                  #; (wrb) s8  <-- 0
#; unknown function (start.S:36)
#;   mv s9, x0
            188000    0x80000060 li      s9, 0                  #; (wrb) s9  <-- 0
#; unknown function (start.S:37)
#;   mv s10, x0
            189000    0x80000064 li      s10, 0                 #; (wrb) s10 <-- 0
#; unknown function (start.S:38)
#;   mv s11, x0
            190000    0x80000068 li      s11, 0                 #; (wrb) s11 <-- 0
#; unknown function (start.S:44)
#;   csrr    t0, misa
            191000    0x8000006c csrr    t0, misa               #; misa = 0x40801129, (wrb) t0  <-- 0x40801129
#; unknown function (start.S:45)
#;   andi    t0, t0, (1 << 3) | (1 << 5) # D/F - single/double precision float extension
            192000    0x80000070 andi    t0, t0, 40             #; t0  = 0x40801129, (wrb) t0  <-- 40
#; unknown function (start.S:46)
#;   beqz    t0, 3f
            193000    0x80000074 beqz    t0, pc + 132           #; t0  = 40, not taken
#; unknown function (start.S:48)
#;   fcvt.d.w f0, zero
            195000    0x80000078 fcvt.d.w ft0, zero             #; ac1  = 0
#; unknown function (start.S:49)
#;   fcvt.d.w f1, zero
            196000    0x8000007c fcvt.d.w ft1, zero             #; ac1  = 0, (f:fpu) ft0  <-- 0.0
            197000                                              #; (f:fpu) ft1  <-- 0.0
#; unknown function (start.S:50)
#;   fcvt.d.w f2, zero
            201000    0x80000080 fcvt.d.w ft2, zero             #; ac1  = 0
#; unknown function (start.S:51)
#;   fcvt.d.w f3, zero
            202000    0x80000084 fcvt.d.w ft3, zero             #; ac1  = 0, (f:fpu) ft2  <-- 0.0
#; unknown function (start.S:52)
#;   fcvt.d.w f4, zero
            203000    0x80000088 fcvt.d.w ft4, zero             #; ac1  = 0, (f:fpu) ft3  <-- 0.0
#; unknown function (start.S:53)
#;   fcvt.d.w f5, zero
            204000    0x8000008c fcvt.d.w ft5, zero             #; ac1  = 0, (f:fpu) ft4  <-- 0.0
#; unknown function (start.S:54)
#;   fcvt.d.w f6, zero
            205000    0x80000090 fcvt.d.w ft6, zero             #; ac1  = 0, (f:fpu) ft5  <-- 0.0
#; unknown function (start.S:55)
#;   fcvt.d.w f7, zero
            206000    0x80000094 fcvt.d.w ft7, zero             #; ac1  = 0, (f:fpu) ft6  <-- 0.0
#; unknown function (start.S:56)
#;   fcvt.d.w f8, zero
            207000    0x80000098 fcvt.d.w fs0, zero             #; ac1  = 0, (f:fpu) ft7  <-- 0.0
#; unknown function (start.S:57)
#;   fcvt.d.w f9, zero
            208000    0x8000009c fcvt.d.w fs1, zero             #; ac1  = 0, (f:fpu) fs0  <-- 0.0
            209000                                              #; (f:fpu) fs1  <-- 0.0
#; unknown function (start.S:58)
#;   fcvt.d.w f10, zero
            213000    0x800000a0 fcvt.d.w fa0, zero             #; ac1  = 0
#; unknown function (start.S:59)
#;   fcvt.d.w f11, zero
            214000    0x800000a4 fcvt.d.w fa1, zero             #; ac1  = 0, (f:fpu) fa0  <-- 0.0
#; unknown function (start.S:60)
#;   fcvt.d.w f12, zero
            215000    0x800000a8 fcvt.d.w fa2, zero             #; ac1  = 0, (f:fpu) fa1  <-- 0.0
#; unknown function (start.S:61)
#;   fcvt.d.w f13, zero
            216000    0x800000ac fcvt.d.w fa3, zero             #; ac1  = 0, (f:fpu) fa2  <-- 0.0
#; unknown function (start.S:62)
#;   fcvt.d.w f14, zero
            217000    0x800000b0 fcvt.d.w fa4, zero             #; ac1  = 0, (f:fpu) fa3  <-- 0.0
#; unknown function (start.S:63)
#;   fcvt.d.w f15, zero
            218000    0x800000b4 fcvt.d.w fa5, zero             #; ac1  = 0, (f:fpu) fa4  <-- 0.0
#; unknown function (start.S:64)
#;   fcvt.d.w f16, zero
            219000    0x800000b8 fcvt.d.w fa6, zero             #; ac1  = 0, (f:fpu) fa5  <-- 0.0
#; unknown function (start.S:65)
#;   fcvt.d.w f17, zero
            220000    0x800000bc fcvt.d.w fa7, zero             #; ac1  = 0, (f:fpu) fa6  <-- 0.0
            221000                                              #; (f:fpu) fa7  <-- 0.0
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
            248000    0x80000100 csrr    a0, mhartid            #; mhartid = 7, (wrb) a0  <-- 7
#; unknown function (start.S:99)
#;   li   t0, SNRT_BASE_HARTID
            249000    0x80000104 li      t0, 0                  #; (wrb) t0  <-- 0
#; unknown function (start.S:100)
#;   sub  a0, a0, t0
            250000    0x80000108 sub     a0, a0, t0             #; a0  = 7, t0  = 0, (wrb) a0  <-- 7
#; unknown function (start.S:101)
#;   li   a1, SNRT_CLUSTER_CORE_NUM
            251000    0x8000010c li      a1, 9                  #; (wrb) a1  <-- 9
#; unknown function (start.S:102)
#;   div  t0, a0, a1
            252000    0x80000110 div     t0, a0, a1             #; a0  = 7, a1  = 9
#; unknown function (start.S:105)
#;   remu a0, a0, a1
            253000    0x80000114 remu    a0, a0, a1             #; a0  = 7, a1  = 9
#; unknown function (start.S:108)
#;   li   a2, SNRT_TCDM_START_ADDR
            254000    0x80000118 lui     a2, 0x10000            #; (wrb) a2  <-- 0x10000000
#; unknown function (start.S:109)
#;   li   t1, SNRT_CLUSTER_OFFSET
            255000    0x8000011c li      t1, 0                  #; (wrb) t1  <-- 0
            269000                                              #; (acc) t0  <-- 0
#; unknown function (start.S:110)
#;   mul  t0, t1, t0
            270000    0x80000120 mul     t0, t1, t0             #; t1  = 0, t0  = 0
            288000                                              #; (acc) a0  <-- 7
            299000                                              #; (acc) t0  <-- 0
#; unknown function (start.S:111)
#;   add  a2, a2, t0
            300000    0x80000124 add     a2, a2, t0             #; a2  = 0x10000000, t0  = 0, (wrb) a2  <-- 0x10000000
#; unknown function (start.S:114)
#;   li   t0, SNRT_TCDM_SIZE
            301000    0x80000128 lui     t0, 0x20               #; (wrb) t0  <-- 0x00020000
#; unknown function (start.S:115)
#;   add  a2, a2, t0
            302000    0x8000012c add     a2, a2, t0             #; a2  = 0x10000000, t0  = 0x00020000, (wrb) a2  <-- 0x10020000
#; unknown function (start.S:121)
#;   la        t0, __cdata_end
            303000    0x80000130 auipc   t0, 0x7                #; (wrb) t0  <-- 0x80007130
            304000    0x80000134 addi    t0, t0, -1176          #; t0  = 0x80007130, (wrb) t0  <-- 0x80006c98
#; unknown function (start.S:122)
#;   la        t1, __cdata_start
            305000    0x80000138 auipc   t1, 0x7                #; (wrb) t1  <-- 0x80007138
            306000    0x8000013c addi    t1, t1, -1184          #; t1  = 0x80007138, (wrb) t1  <-- 0x80006c98
#; unknown function (start.S:123)
#;   sub       t0, t0, t1
            307000    0x80000140 sub     t0, t0, t1             #; t0  = 0x80006c98, t1  = 0x80006c98, (wrb) t0  <-- 0
#; unknown function (start.S:124)
#;   sub       a2, a2, t0
            308000    0x80000144 sub     a2, a2, t0             #; a2  = 0x10020000, t0  = 0, (wrb) a2  <-- 0x10020000
#; unknown function (start.S:125)
#;   la        t0, __cbss_end
            309000    0x80000148 auipc   t0, 0x7                #; (wrb) t0  <-- 0x80007148
            310000    0x8000014c addi    t0, t0, -1184          #; t0  = 0x80007148, (wrb) t0  <-- 0x80006ca8
#; unknown function (start.S:126)
#;   la        t1, __cbss_start
            311000    0x80000150 auipc   t1, 0x7                #; (wrb) t1  <-- 0x80007150
            312000    0x80000154 addi    t1, t1, -1208          #; t1  = 0x80007150, (wrb) t1  <-- 0x80006c98
#; unknown function (start.S:127)
#;   sub       t0, t0, t1
            313000    0x80000158 sub     t0, t0, t1             #; t0  = 0x80006ca8, t1  = 0x80006c98, (wrb) t0  <-- 16
#; unknown function (start.S:128)
#;   sub       a2, a2, t0
            314000    0x8000015c sub     a2, a2, t0             #; a2  = 0x10020000, t0  = 16, (wrb) a2  <-- 0x1001fff0
#; unknown function (start.S:135)
#;   addi      a2, a2, -8
            315000    0x80000160 addi    a2, a2, -8             #; a2  = 0x1001fff0, (wrb) a2  <-- 0x1001ffe8
#; unknown function (start.S:136)
#;   sw        zero, 0(a2)
            316000    0x80000164 sw      zero, 0(a2)            #; a2  = 0x1001ffe8, 0 ~~> Word[0x1001ffe8]
#; unknown function (start.S:140)
#;   sll       t0, a0, SNRT_LOG2_STACK_SIZE
            317000    0x80000168 slli    t0, a0, 10             #; a0  = 7, (wrb) t0  <-- 7168
#; unknown function (start.S:143)
#;   sub       sp, a2, t0
            318000    0x8000016c sub     sp, a2, t0             #; a2  = 0x1001ffe8, t0  = 7168, (wrb) sp  <-- 0x1001e3e8
#; unknown function (start.S:147)
#;   sll       t1, a1, t2
            319000    0x80000170 sll     t1, a1, t2             #; a1  = 9, t2  = 0, (wrb) t1  <-- 9
#; unknown function (start.S:148)
#;   sub       a2, a2, t1
            320000    0x80000174 sub     a2, a2, t1             #; a2  = 0x1001ffe8, t1  = 9, (wrb) a2  <-- 0x1001ffdf
#; unknown function (start.S:151)
#;   slli      t0, a0, 3  # this hart
            321000    0x80000178 slli    t0, a0, 3              #; a0  = 7, (wrb) t0  <-- 56
#; unknown function (start.S:152)
#;   slli      t1, a1, 3  # all harts
            322000    0x8000017c slli    t1, a1, 3              #; a1  = 9, (wrb) t1  <-- 72
#; unknown function (start.S:153)
#;   sub       sp, sp, t0
            324000    0x80000180 sub     sp, sp, t0             #; sp  = 0x1001e3e8, t0  = 56, (wrb) sp  <-- 0x1001e3b0
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
            331000    0x8000019c sub     sp, sp, t0             #; sp  = 0x1001e3b0, t0  = 0, (wrb) sp  <-- 0x1001e3b0
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
            341000    0x800001b4 sub     sp, sp, t0             #; sp  = 0x1001e3b0, t0  = 16, (wrb) sp  <-- 0x1001e3a0
#; unknown function (start.S:168)
#;   andi      sp, sp, ~0x7 # align to 8B
            342000    0x800001b8 andi    sp, sp, -8             #; sp  = 0x1001e3a0, (wrb) sp  <-- 0x1001e3a0
#; unknown function (start.S:170)
#;   mv        tp, sp
            343000    0x800001bc mv      tp, sp                 #; sp  = 0x1001e3a0, (wrb) tp  <-- 0x1001e3a0
#; unknown function (start.S:172)
#;   andi      sp, sp, ~0x7 # align stack to 8B
            348000    0x800001c0 andi    sp, sp, -8             #; sp  = 0x1001e3a0, (wrb) sp  <-- 0x1001e3a0
#; unknown function (start.S:183)
#;   call snrt_main
            349000    0x800001c4 auipc   ra, 0x6                #; (wrb) ra  <-- 0x800061c4
            350000    0x800001c8 jalr    ra, ra, -796           #; ra  = 0x800061c4, (wrb) ra  <-- 0x800001cc, goto 0x80005ea8
#; snrt_main (start.c:106)
#;   void snrt_main() {
            362000    0x80005ea8 addi    sp, sp, -32            #; sp  = 0x1001e3a0, (wrb) sp  <-- 0x1001e380
#; snrt_main (start.c:-1)
#; 
            363000    0x80005eac sw      ra, 28(sp)             #; sp  = 0x1001e380, 0x800001cc ~~> Word[0x1001e39c]
            364000    0x80005eb0 sw      s0, 24(sp)             #; sp  = 0x1001e380, 0 ~~> Word[0x1001e398]
            365000    0x80005eb4 sw      s1, 20(sp)             #; sp  = 0x1001e380, 0 ~~> Word[0x1001e394]
            366000    0x80005eb8 sw      s2, 16(sp)             #; sp  = 0x1001e380, 0 ~~> Word[0x1001e390]
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:29)
#;     snrt_is_dm_core (team.h:57)
#;       snrt_is_compute_core (team.h:53)
#;         snrt_cluster_core_idx (team.h:41)
#;           snrt_global_core_idx (team.h:28)
#;             snrt_hartid (team.h:7)
#;               asm("csrr %0, mhartid" : "=r"(hartid));
            367000    0x80005ebc csrr    s2, mhartid            #; mhartid = 7, (wrb) s2  <-- 7
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
            376000    0x80005ec8 mulhu   a0, s2, a0             #; s2  = 7, a0  = 0x38e38e39
            387000                                              #; (acc) a0  <-- 1
            388000    0x80005ecc srli    a0, a0, 1              #; a0  = 1, (wrb) a0  <-- 0
            389000    0x80005ed0 slli    a1, a0, 3              #; a0  = 0, (wrb) a1  <-- 0
            390000    0x80005ed4 add     a0, a1, a0             #; a1  = 0, a0  = 0, (wrb) a0  <-- 0
            391000    0x80005ed8 sub     a5, s2, a0             #; s2  = 7, a0  = 0, (wrb) a5  <-- 7
            392000    0x80005edc li      a6, 8                  #; (wrb) a6  <-- 8
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:29)
#;     if (snrt_is_dm_core()) {
            393000    0x80005ee0 bltu    a5, a6, pc + 524       #; a5  = 7, a6  = 8, taken, goto 0x800060ec
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:54)
#;     snrt_cluster_hw_barrier (sync.h:59)
#;       asm volatile("csrr x0, 0x7C2" ::: "memory");
            409000    0x800060ec csrs    unknown_7c2, zero      #; csr@7c2 = 0
#; snrt_main (start.c:114)
#;   snrt_init_tls (start.c:-1)
#; 
            744000    0x800060f0 sltiu   a7, a5, 8              #; a5  = 7, (wrb) a7  <-- 1
#; snrt_main (start.c:122)
#;   snrt_init_bss (start.c:63)
#;     if (snrt_cluster_idx() == 0 && snrt_is_dm_core()) {
            745000    0x800060f4 bltu    a6, s2, pc + 96        #; a6  = 8, s2  = 7, not taken
#; snrt_main (start.c:122)
#;   snrt_init_bss (start.c:63)
#;     snrt_is_dm_core (team.h:57)
#;       snrt_is_compute_core (team.h:53)
#;         snrt_cluster_core_idx (team.h:41)
#;           return snrt_global_core_idx() % snrt_cluster_core_num();
            746000    0x800060f8 andi    a0, s2, 255            #; s2  = 7, (wrb) a0  <-- 7
            747000    0x800060fc lui     a1, 0x38e39            #; (wrb) a1  <-- 0x38e39000
            748000    0x80006100 addi    a1, a1, -455           #; a1  = 0x38e39000, (wrb) a1  <-- 0x38e38e39
            749000    0x80006104 mulhu   a1, a0, a1             #; a0  = 7, a1  = 0x38e38e39
            752000                                              #; (acc) a1  <-- 1
            753000    0x80006108 srli    a1, a1, 1              #; a1  = 1, (wrb) a1  <-- 0
            754000    0x8000610c slli    a2, a1, 3              #; a1  = 0, (wrb) a2  <-- 0
            755000    0x80006110 add     a1, a2, a1             #; a2  = 0, a1  = 0, (wrb) a1  <-- 0
            756000    0x80006114 sub     a0, a0, a1             #; a0  = 7, a1  = 0, (wrb) a0  <-- 7
#; snrt_main (start.c:122)
#;   snrt_init_bss (start.c:63)
#;     snrt_is_dm_core (team.h:57)
#;       snrt_is_compute_core (team.h:53)
#;         return snrt_cluster_core_idx() < snrt_cluster_compute_core_num();
            757000    0x80006118 sltiu   a1, a0, 8              #; a0  = 7, (wrb) a1  <-- 1
#; snrt_main (start.c:122)
#;   snrt_init_bss (start.c:63)
#;     if (snrt_cluster_idx() == 0 && snrt_is_dm_core()) {
            758000    0x8000611c auipc   a0, 0x1                #; (wrb) a0  <-- 0x8000711c
            760000    0x80006120 addi    a0, a0, -292           #; a0  = 0x8000711c, (wrb) a0  <-- 0x80006ff8
            761000    0x80006124 auipc   a2, 0x1                #; (wrb) a2  <-- 0x80007124
            762000    0x80006128 addi    a2, a2, 512            #; a2  = 0x80007124, (wrb) a2  <-- 0x80007324
            763000    0x8000612c sub     a4, a2, a0             #; a2  = 0x80007324, a0  = 0x80006ff8, (wrb) a4  <-- 812
            764000    0x80006130 seqz    a2, a4                 #; a4  = 812, (wrb) a2  <-- 0
            765000    0x80006134 or      a1, a1, a2             #; a1  = 1, a2  = 0, (wrb) a1  <-- 1
            766000    0x80006138 bnez    a1, pc + 28            #; a1  = 1, taken, goto 0x80006154
#; snrt_main (start.c:130)
#;   snrt_init_cls (start.c:76)
#;     _cls_ptr = (cls_t*)snrt_cls_base_addr();
            772000    0x80006154 auipc   a5, 0x1                #; (wrb) a5  <-- 0x80007154
            773000    0x80006158 addi    a5, a5, -1212          #; a5  = 0x80007154, (wrb) a5  <-- 0x80006c98
            774000    0x8000615c auipc   t0, 0x1                #; (wrb) t0  <-- 0x8000715c
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
            799000    0x8000618c add     a2, a2, tp             #; a2  = 0, tp  = 0x1001e3a0, (wrb) a2  <-- 0x1001e3a0
            800000    0x80006190 sw      a1, 0(a2)              #; a2  = 0x1001e3a0, 0x1001fff0 ~~> Word[0x1001e3a0]
#; snrt_main (start.c:130)
#;   snrt_init_cls (start.c:79)
#;     if (snrt_is_dm_core()) {
            801000    0x80006194 bnez    a7, pc + 172           #; a7  = 1, taken, goto 0x80006240
#; snrt_main (start.c:151)
#;   snrt_cluster_hw_barrier (sync.h:59)
#;     asm volatile("csrr x0, 0x7C2" ::: "memory");
            821000    0x80006240 csrs    unknown_7c2, zero      #; csr@7c2 = 0
#; snrt_main (start.c:160)
#;   exit_code = main();
            898000    0x80006244 auipc   ra, 0xffffc            #; (wrb) ra  <-- 0x80002244
            899000    0x80006248 jalr    ra, ra, -448           #; ra  = 0x80002244, (wrb) ra  <-- 0x8000624c, goto 0x80002084
#; main (main.c:9)
#;   int main() {
            911000    0x80002084 addi    sp, sp, -48            #; sp  = 0x1001e380, (wrb) sp  <-- 0x1001e350
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:-1)
#; 
            912000    0x80002088 sw      s0, 44(sp)             #; sp  = 0x1001e350, 0x80006ca8 ~~> Word[0x1001e37c]
            913000    0x8000208c sw      s1, 40(sp)             #; sp  = 0x1001e350, 0x80006c98 ~~> Word[0x1001e378]
            914000    0x80002090 sw      s2, 36(sp)             #; sp  = 0x1001e350, 7 ~~> Word[0x1001e374]
            915000    0x80002094 sw      s3, 32(sp)             #; sp  = 0x1001e350, 0 ~~> Word[0x1001e370]
            916000    0x80002098 sw      s4, 28(sp)             #; sp  = 0x1001e350, 0 ~~> Word[0x1001e36c]
            917000    0x8000209c sw      s5, 24(sp)             #; sp  = 0x1001e350, 0 ~~> Word[0x1001e368]
            923000    0x800020a0 sw      s6, 20(sp)             #; sp  = 0x1001e350, 0 ~~> Word[0x1001e364]
            924000    0x800020a4 sw      s7, 16(sp)             #; sp  = 0x1001e350, 0 ~~> Word[0x1001e360]
            925000    0x800020a8 sw      s8, 12(sp)             #; sp  = 0x1001e350, 0 ~~> Word[0x1001e35c]
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:815)
#;     snrt_cluster_core_idx (team.h:41)
#;       snrt_global_core_idx (team.h:28)
#;         snrt_hartid (team.h:7)
#;           asm("csrr %0, mhartid" : "=r"(hartid));
            926000    0x800020ac csrr    t2, mhartid            #; mhartid = 7, (wrb) t2  <-- 7
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
            929000    0x800020b8 mulhu   a0, t2, a0             #; t2  = 7, a0  = 0x38e38e39
            933000                                              #; (acc) a0  <-- 1
            934000    0x800020bc srli    a0, a0, 1              #; a0  = 1, (wrb) a0  <-- 0
            935000    0x800020c0 slli    a1, a0, 3              #; a0  = 0, (wrb) a1  <-- 0
            936000    0x800020c4 add     a0, a1, a0             #; a1  = 0, a0  = 0, (wrb) a0  <-- 0
#; main (main.c:10)
#;   batchnorm_backward_training (team.h:-1)
#; 
            937000    0x800020c8 auipc   t3, 0x5                #; (wrb) t3  <-- 0x800070c8
            938000    0x800020cc addi    t3, t3, -360           #; t3  = 0x800070c8, (wrb) t3  <-- 0x80006f60
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:819)
#;     uint32_t H = l->IH;
            939000    0x800020d0 lw      a7, 4(t3)              #; t3  = 0x80006f60, a7  <~~ Word[0x80006f64]
            951000                                              #; (lsu) a7  <-- 2
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:820)
#;     uint32_t W = l->IW;
            952000    0x800020d4 lw      t0, 8(t3)              #; t3  = 0x80006f60, t0  <~~ Word[0x80006f68]
           1001000                                              #; (lsu) t0  <-- 2
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:821)
#;     uint32_t C = l->CI;
           1002000    0x800020d8 lw      s2, 0(t3)              #; t3  = 0x80006f60, s2  <~~ Word[0x80006f60]
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:875)
#;     snrt_mcycle (riscv.h:17)
#;       asm volatile("csrr %0, mcycle" : "=r"(r) : : "memory");
           1003000    0x800020dc csrr    a1, mcycle             #; mcycle = 999, (wrb) a1  <-- 999
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:815)
#;     snrt_cluster_core_idx (team.h:41)
#;       return snrt_global_core_idx() % snrt_cluster_core_num();
           1004000    0x800020e0 sub     s4, t2, a0             #; t2  = 7, a0  = 0, (wrb) s4  <-- 7
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:815)
#;     snrt_cluster_core_idx (team.h:-1)
#; 
           1005000    0x800020e4 li      a0, 8                  #; (wrb) a0  <-- 8
           1050000                                              #; (lsu) s2  <-- 4
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:835)
#;     ptr += C;
           1051000    0x800020e8 slli    a4, s2, 3              #; s2  = 4, (wrb) a4  <-- 32
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:876)
#;     if (snrt_is_dm_core()) {
           1052000    0x800020ec bgeu    s4, a0, pc + 340       #; s4  = 7, a0  = 8, not taken
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:-1)
#; 
           1053000    0x800020f0 srli    a0, s2, 3              #; s2  = 4, (wrb) a0  <-- 0
           1054000    0x800020f4 andi    a1, s2, 7              #; s2  = 4, (wrb) a1  <-- 4
           1055000    0x800020f8 sltu    a1, s4, a1             #; s4  = 7, a1  = 4, (wrb) a1  <-- 0
           1056000    0x800020fc add     a0, a0, a1             #; a0  = 0, a1  = 0, (wrb) a0  <-- 0
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:882)
#;     snrt_ssr_loop_1d (ssr.h:74)
#;       --b0;
           1057000    0x80002100 addi    a1, a0, -1             #; a0  = 0, (wrb) a1  <-- -1
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:882)
#;     snrt_ssr_loop_1d (ssr.h:75)
#;       write_ssr_cfg (ssr.h:68)
#;         asm volatile("scfgwi %[value], %[dm] | %[reg]<<5\n" ::[value] "r"(value),
           1058000    0x80002104 scfgwi  a1, 95                 #; a1  = -1
           1059000    0x80002108 li      a2, 64                 #; (wrb) a2  <-- 64
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:882)
#;     snrt_ssr_loop_1d (ssr.h:77)
#;       write_ssr_cfg (ssr.h:68)
#;         asm volatile("scfgwi %[value], %[dm] | %[reg]<<5\n" ::[value] "r"(value),
           1060000    0x8000210c scfgwi  a2, 223                #; a2  = 64
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:885)
#;     snrt_mcycle (riscv.h:17)
#;       asm volatile("csrr %0, mcycle" : "=r"(r) : : "memory");
           1061000    0x80002110 csrr    a2, mcycle             #; mcycle = 1057, (wrb) a2  <-- 1057
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:886)
#;     snrt_cluster_hw_barrier();
           1062000    0x80002114 csrwi   unknown_7c3, 0         #; 
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:887)
#;     snrt_cluster_hw_barrier (sync.h:59)
#;       asm volatile("csrr x0, 0x7C2" ::: "memory");
           1063000    0x80002118 csrs    unknown_7c2, zero      #; csr@7c2 = 0
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:895)
#;     snrt_mcycle (riscv.h:17)
#;       asm volatile("csrr %0, mcycle" : "=r"(r) : : "memory");
           1178000    0x8000211c csrr    a2, mcycle             #; mcycle = 1174, (wrb) a2  <-- 1174
#; main (main.c:10)
#;   batchnorm_backward_training (batchnorm.h:896)
#;     snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, &invstd[compute_id]);
           1179000    0x80002120 beqz    a0, pc + 1196          #; a0  = 0, taken, goto 0x800025cc
#; main (main.c:12)
#;   snrt_global_barrier (sync.h:81)
#;     snrt_cluster_hw_barrier (sync.h:59)
#;       asm volatile("csrr x0, 0x7C2" ::: "memory");
           1193000    0x800025cc csrs    unknown_7c2, zero      #; csr@7c2 = 0
#; main (main.c:14)
#;   return 0;
           1285000    0x800025d0 li      a0, 0                  #; (wrb) a0  <-- 0
           1286000    0x800025d4 lw      s8, 12(sp)             #; sp  = 0x1001e350, s8  <~~ Word[0x1001e35c]
           1289000                                              #; (lsu) s8  <-- 0
           1290000    0x800025d8 lw      s7, 16(sp)             #; sp  = 0x1001e350, s7  <~~ Word[0x1001e360]
           1293000                                              #; (lsu) s7  <-- 0
           1294000    0x800025dc lw      s6, 20(sp)             #; sp  = 0x1001e350, s6  <~~ Word[0x1001e364]
           1297000                                              #; (lsu) s6  <-- 0
           1298000    0x800025e0 lw      s5, 24(sp)             #; sp  = 0x1001e350, s5  <~~ Word[0x1001e368]
           1301000                                              #; (lsu) s5  <-- 0
           1302000    0x800025e4 lw      s4, 28(sp)             #; sp  = 0x1001e350, s4  <~~ Word[0x1001e36c]
           1305000                                              #; (lsu) s4  <-- 0
           1306000    0x800025e8 lw      s3, 32(sp)             #; sp  = 0x1001e350, s3  <~~ Word[0x1001e370]
           1309000                                              #; (lsu) s3  <-- 0
           1310000    0x800025ec lw      s2, 36(sp)             #; sp  = 0x1001e350, s2  <~~ Word[0x1001e374]
           1313000                                              #; (lsu) s2  <-- 7
           1314000    0x800025f0 lw      s1, 40(sp)             #; sp  = 0x1001e350, s1  <~~ Word[0x1001e378]
           1317000                                              #; (lsu) s1  <-- 0x80006c98
