// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0


double ifmap[1][8][8][1];

double ofmap[1][4][4][1];

maxpool_layer_t layer = {
	.CO = 1,
	.CI = 1,
	.IH = 8,
	.IW = 8,
	.OH = 4,
	.OW = 4,
	.FH = 2,
	.FW = 2,
	.tile_ci = 1,
	.ifmap = ifmap,
	.ofmap = ofmap
};

double ifmap[1][8][8][1] = {
	0.2995562877502637,
	-0.3402723143550538,
	-0.7331361007165538,
	0.24904992357977876,
	0.4780347016744464,
	-0.9954168307888536,
	0.8239191817788585,
	-0.018728729244763993,
	0.13346330686979527,
	0.18996944175263808,
	0.0807615803099693,
	0.41946015140529674,
	-1.3221584274660605,
	0.46700278861928723,
	0.8445823549282497,
	1.3019340290708845,
	1.8448885093030065,
	-2.0773600718526932,
	-0.14883240827467462,
	-1.2752325216668503,
	2.8536471208979433,
	-0.07729526178752644,
	-1.6099870201619317,
	0.5939979568631141,
	2.8510918408680124,
	-1.5043005890504582,
	0.4455096569146938,
	0.08093711755806289,
	0.21407749737048995,
	-0.7063829736735863,
	-1.3941268553154642,
	-0.05334598793633222,
	0.24237779080894106,
	0.9906934705391687,
	0.02971415623949211,
	-0.5595901539006848,
	0.9340946495293597,
	-0.7025528681323762,
	1.0202696745617659,
	0.056387506785403815,
	1.4238494481371482,
	0.603530025130453,
	0.5077333887313688,
	-1.02228304173901,
	1.0841297942511154,
	0.6600059658549468,
	0.5529895567096951,
	0.7323856221618156,
	0.572007132100381,
	0.5102128857829208,
	2.0486968982261202,
	-0.44546162876683887,
	1.3032022234889244,
	-0.4820520883676177,
	0.6146813355739139,
	0.571492357564412,
	-0.7318379837818061,
	0.7372524407400997,
	-1.5530036453685645,
	-2.256777082059624,
	-0.41165660712793767,
	0.08317733846151737,
	0.5706575386796322,
	0.10738590560666669,
	0.28320739441795323,
};

#ifdef BIST
double golden[1][4][4][32] = {
	0.2995562877502637,
	0.9906934705391687,
	0.07767737996636266,
	1.2160699112636764,
	0.9340946495293597,
	1.548288167937,
	1.0202696745617659,
	1.3163800108985646,
	1.4238494481371482,
	0.6530495473603715,
	0.5077333887313688,
	0.41946015140529674,
	2.0712307224954674,
	0.6600059658549468,
	1.44017146059469,
	1.3019340290708845,
	1.8448885093030065,
	0.5102128857829208,
	2.0486968982261202,
	-0.44546162876683887,
	2.8536471208979433,
	0.9077850533771569,
	0.6146813355739139,
	1.3958505210549053,
	2.8510918408680124,
	0.7372524407400997,
	0.4455096569146938,
	0.12572172650108407,
	0.23609553256542695,
	0.08317733846151737,
	0.5706575386796322,
	0.37108183095020375,
	0.4285173645433462,
	2.1413916159323136,
	1.981160746670956,
	0.8956621506501764,
	0.08511336387296158,
	0.7739177723108309,
	1.3346481596559676,
	1.25609380794485,
	0.9068876194863097,
	1.9579409009069817,
	1.5824852577508077,
	0.07037045527138296,
	1.1174296841710698,
	2.5064779220476376,
	0.997991850817377,
	1.0763690216500268,
	1.423932009681719,
	1.0664992078500592,
	1.993616236752697,
	1.0636457383418108,
	0.4760756036980966,
	0.2721281447061126,
	1.0232681185457035,
	1.2838646589587073,
	0.8118142867300344,
	0.37761170331094157,
	1.9679440499519993,
	1.2473531605780193,
	1.371312887239788,
	0.49835413266819356,
	0.7831976119427991,
	0.7476571054869932,
	0.7817673742546976,
	-0.3466771947976861,
	0.9635794871121548,
	1.6459648576014418,
	1.3917635799868133,
	0.17859923880951847,
	0.7611949933670529,
	-0.3443663683433915,
	2.7190025950526526,
	0.859036998403129,
	0.4716519250182116,
	1.1236395260389371,
	0.43438863938401984,
	0.5143771232291072,
	1.5886336771168668,
	0.9847648224117718,
	1.247922968974058,
	0.34215369871376844,
	1.8548042098231963,
	0.8829476857079316,
	1.199426224041857,
	-0.22291945275506458,
	0.9035218354603602,
	0.0125526247658802,
	0.43758672808479837,
	0.2636875321138984,
	0.5125620452253108,
	0.49295286102524044,
	0.2318653380046876,
	0.64706666019578,
	1.1971503331114541,
	1.1869006875492747,
	1.7342104609172728,
	2.5161385989502594,
	2.1301275306235565,
	0.6518019493570052,
	0.9478581977287968,
	0.9425485271043043,
	0.2730005353024221,
	1.313122245567395,
	0.3073860300996402,
	1.8807687476094996,
	1.756443180184832,
	0.8178069723083835,
	0.24943584530698598,
	0.08667290592050396,
	1.1696801481884702,
	0.8900182108394689,
	1.0390897743423966,
	1.1953652639556394,
	2.057751677950784,
	1.1538879916827052,
	0.3503211304018375,
	1.415842532715168,
	1.4723540101508836,
	1.7267743114344978,
	0.8679359705510734,
	1.1244580271586777,
	0.835126803441082,
	-0.3494884497630849,
	1.7562161679995132,
	0.689824201173873,
	0.7916834209392376,
	0.9914820089613138,
	1.504572275678796,
	1.417476842717687,
	0.7418633974455897,
	0.46742788122395995,
	0.8505308967944145,
	2.177326020841736,
	1.0157942265133257,
	1.5109902204218253,
	0.9755465820040533,
	1.1510677794844635,
	1.2593028322942528,
	1.457813522969226,
	2.0503199405113706,
	0.7093874142064377,
	1.1334557143017412,
	1.3946491399984025,
	1.9951334816061994,
	0.899919260122491,
	1.6134203350119927,
	0.9994754870230482,
	0.07734968673664687,
	1.2098862819236005,
	1.1617475981837795,
	2.264358935950376,
	1.5065733412421733,
	3.4237712468503103,
	0.27932962145356316,
	1.7611911540817928,
	1.181971977987871,
	1.524711206099055,
	1.160481352505748,
	1.8409181821652891,
	1.2715909667771088,
	1.3602887543809026,
	-0.1504751849288266,
	0.5800148267422183,
	0.6772607869261693,
	0.19368409881631005,
	1.6129488803273395,
	-0.062040923189698165,
	-0.1212982442332069,
	-0.1530695247591354,
	2.207695578155158,
	1.8372706762077982,
	1.850491258358576,
	0.2672756574827596,
	1.2916562025813139,
	0.7388050176954863,
	1.862246649641681,
	2.0182530686564895,
	-0.09854463425203973,
	1.22745042848715,
	1.8473014772988974,
	0.945043974186398,
	0.5665224930728516,
	0.8149334833372205,
	0.25697771346841397,
	0.6819214512912009,
	1.427347151863727,
	-0.13454483357874214,
	1.0392171119322082,
	1.2284263304913572,
	1.8273660232377136,
	0.352400820758953,
	1.108461181451433,
	1.7962695897011005,
	0.4319095829234074,
	0.675401077688519,
	1.9920454844101763,
	1.8110431275164358,
	0.6164470190315824,
	1.7075504892947724,
	0.40810561816935464,
	0.542636780326917,
	0.6951962730289927,
	1.0847219277436837,
	0.6391454162969747,
	2.048440256292676,
	1.05805011001695,
	-0.25332197648462423,
	1.1127961330998304,
	0.8203366802620794,
	2.395067138296113,
	0.002668556331391873,
	1.4335843287274053,
	1.0045993947511778,
	1.0788518344074862,
	1.6332065112971001,
	0.707300443390559,
	1.6425500385797525,
	0.2439549047998887,
	1.2706120469339364,
	1.8371154985886862,
	0.5071009399593757,
	1.688961038095004,
	1.6482773735238936,
	0.8651734647119074,
	1.53173972311803,
	0.9096040699613742,
	1.8141622548156382,
	1.498700405481595,
	1.029960808199997,
	0.42155963063098567,
	1.5535736491676595,
	2.080423035876765,
	1.5308606521156258,
	1.2736538935419954,
	0.5734589140951897,
	1.3293513220384825,
	0.49482862922777465,
	2.4488104808936493,
	0.8305024600077695,
	2.681801981114757,
	1.7752306448722606,
	0.5680057504067159,
	0.4874774972013086,
	0.9547242579775002,
	1.4850546815331578,
	2.207851490608375,
	0.8352543400425974,
	0.2108448776087118,
	0.3229857369820825,
	-0.4453218737326093,
	1.59925697493435,
	0.19313774990116925,
	0.7479073753397237,
	0.3469301281229362,
	1.38335867526748,
	0.6551143052731048,
	-0.4234650258705706,
	2.059923212244384,
	0.002840677025019378,
	1.797754234791071,
	0.43343311885236446,
	0.8545701717376761,
	2.1112388578012884,
	2.1497158382519235,
	3.0843102220187633,
	0.6622884149541679,
	0.8833144877181079,
	0.9541627279264773,
	1.4382328249156382,
	0.05819716648813362,
	0.7629575394167195,
	0.622613137075688,
	0.6116700031464317,
	1.744020696757244,
	0.4763931829831273,
	1.0259687873673153,
	2.90124785320579,
	0.8295084980828852,
	1.4876059940278459,
	0.7768976641774541,
	0.809667872762421,
	1.4533777876325222,
	0.5675579810614343,
	0.5053866496884525,
	0.6397252376986503,
	0.4808356602618045,
	0.6485877172023825,
	1.3109621860249783,
	1.8754720735908623,
	0.7751228541207369,
	1.0066107886868416,
	0.29946965798993297,
	0.6955788134794414,
	1.980986216379721,
	1.9700777629678679,
	0.8508105491449758,
	0.9671035281600022,
	1.4938404772180522,
	2.6779598810353225,
	1.1690740710716336,
	2.315456482213615,
	0.8348604945110601,
	0.2141878125426338,
	1.7456452297639542,
	-0.5606885258433681,
	1.9280259690970352,
	0.420689070376205,
	0.48193063531644537,
	1.4109003071198862,
	1.4448964367746397,
	0.6269276746157323,
	0.8524501556338853,
	1.749809215983805,
	0.5810375830537511,
	1.9203285366540146,
	0.22201020504142566,
	0.7465865033369634,
	1.1907305012760756,
	-0.2757466733288484,
	1.9312142505772192,
	-0.4534667104093323,
	1.0133904557056803,
	2.1807319429357963,
	1.452604901391212,
	0.6099936154303692,
	0.011103897250099562,
	1.3299880805922875,
	1.4827558216775214,
	1.9106686856645196,
	1.8174697527666106,
	0.36248256904087583,
	1.2954089803971967,
	2.179937643276768,
	1.3277786760361687,
	0.7634824898379703,
	1.010839700197812,
	2.327922240143418,
	0.9031672530350638,
	0.6054083058374654,
	0.8136749201194545,
	1.9987164641082087,
	-0.27202686813738103,
	2.975492133500898,
	0.4922810729687261,
	0.15383957509758237,
	0.0754147254586834,
	0.8421602450844566,
	1.4399368526678278,
	1.1788921695577175,
	2.532325541632594,
	0.18139626346512772,
	0.8636024370428342,
	1.1519037944728634,
	2.425005496346911,
	1.54663939130953,
	0.6929251584566181,
	1.090316288383937,
	1.0111021595209633,
	-0.22203877342660364,
	0.4878869471857865,
	2.287956668802193,
	0.7699322703630633,
	0.2580803877827819,
	-0.3913051481757693,
	0.05784031807603419,
	1.4097204921161488,
	0.8544108257491024,
	0.5757151238996234,
	1.0350931612437495,
	2.269233244076525,
	0.7614369056740486,
	1.2824833220745997,
	0.39258330183185886,
	1.1669873661563108,
	-0.6461446779487539,
	0.1619562886451128,
	0.2669263499455465,
	1.109609770945844,
	1.99145358910135,
	1.4512382841053386,
	0.8153426406136252,
	0.7429642513221194,
	0.06348346009811161,
	1.4654350727468453,
	0.23432766680511513,
	0.2863012087770702,
	1.4026313276860414,
	0.23274595474814475,
	2.6380786365181796,
	-0.007188030468140175,
	0.782406872557914,
	0.5654999312912312,
	0.5526482370477976,
	1.688709572678028,
	1.1536049623647049,
	0.3852736787379978,
	0.2706183364237272,
	1.4095581263320807,
	0.7467893991401736,
	1.0306895727537915,
	0.5592128760780101,
	1.4717430631656925,
	1.3420338787636203,
	0.6657305691375492,
	1.185062371327038,
	0.17431258922451653,
	1.535151051934949,
	2.7590525887014667,
	1.2380051766092655,
	3.5953876986672255,
	0.6612001818907256,
	0.31200772222466067,
	1.4420580869170465,
	1.3390674367668436,
	0.9856658198526146,
	1.1145903864961688,
	1.092646154841347,
	1.3612493070253837,
	1.4061442335836205,
	0.8387576576827396,
	0.9803566659279874,
	1.1809578134345797,
	-0.5846446870404606,
	1.5494731988925168,
	1.7607346842800078,
	1.5913375580679756,
	0.5987174412346722,
	0.5243841797548443,
	0.9146548225539167,
	-0.364632760285267,
	0.24044416879470595,
	1.8901581231152924,
	0.536910685665978,
	0.610220453599741,
	1.7703882319843902,
	0.6391954485208522,
	1.0330281295592512,
	1.557358039479812,
	1.5321216571398668,
	0.48707383897255696,
	1.2480509452812811,
	1.0778363941578806,
	1.0421502948740846,
	0.7870016058915985,
	0.39608506375087277,
	1.0076432223870029,
	1.688912150364615,
	0.4428268084752764,
	0.8441868298852333,
	0.9219834856645437,
	1.0504938948460687,
	1.231436269706581,
	1.5545069571528942,
	0.5946658094550371,
	2.253287365111219,
	0.560337204317833,
	0.1616835183072944,
	0.8247851368961774,
	0.9148458058134971,
	0.9001838057649912,
	0.6967400087768053,
	1.0875663945370075,
	1.6279732482605165,
	2.22854455207779,
	2.417337785331285,
	1.3206137582172544,
	-0.35212526236220854,
	0.3469149176826813,
	0.9197089124134338,
	1.429798559996883,
	1.7738534285950363,
	0.2937618607534383,
	1.9521841872543286,
	-0.25957042793288443,
	-0.03287499950025958,
	0.8949404757727989,
	1.1493902335657455,
	0.5698707282163747,
	1.4973252224613436,
	1.1755166198951001,
	2.4477905540112914,
	0.023687715374861383,
	1.6317833840380092,
	0.11041826907812055,
	1.5849730687530486,
	1.1838674589073483,
	2.1818716851528808,
	0.6318144321904159,
	1.269091194574501,
	1.239178450003301,
	-0.14812844625039906,
	0.11228583803533418,
	1.8590918614847836,
	0.1465398461641377,
	0.16284752539836636,
	2.85934273014681,
	1.3299424270339977,
	0.5166078996711738,
	0.42973964778170143,
	2.0063313680095405,
	1.4198792364404926,
	0.8488733195590197,
	0.9892360243254097,
	0.4159405289601604,
	1.001353646981955,
	1.9089401526170708,
	2.979844425070889,
	1.8425900504983932,
	1.1476608122361165,
	1.3123618498586958,
	-0.5546344165085834,
};
#endif // BIST
