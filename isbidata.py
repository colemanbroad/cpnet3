isbi_scales = {

  "Fluo-C3DH-A549":      (0.126, 0.126, 1.0),
  "Fluo-C3DH-H157":      (0.126, 0.126, 0.5),
  "Fluo-C3DL-MDA231":    (1.242, 1.242, 6.0),
  "Fluo-N3DH-CE":        (0.09 , 0.09, 1.0),
  "Fluo-N3DH-CHO":       (0.202, 0.202, 1.0),
  "Fluo-N3DL-DRO":       (0.406, 0.406, 2.03),
  "Fluo-N3DL-TRIC":      (1.,1.,1.), # NA due to cartographic projections
  "Fluo-N3DL-TRIF":      (0.38 , 0.38, 0.38),
  "Fluo-C3DH-A549-SIM":  (0.126, 0.126, 1.0),
  "Fluo-N3DH-SIM+":      (0.125, 0.125, 0.200),

  "BF-C2DL-HSC" :        (0.645 ,0.645),
  "BF-C2DL-MuSC" :       (0.645 ,0.645),
  "DIC-C2DH-HeLa" :      (0.19 ,0.19),
  "Fluo-C2DL-MSC" :      (0.3 ,0.3), # (0.3977 x 0.3977) for dataset 2?,
  "Fluo-N2DH-GOWT1" :    (0.240 ,0.240),
  "Fluo-N2DL-HeLa" :     (0.645 ,0.645),
  "PhC-C2DH-U373" :      (0.65 ,0.65),
  "PhC-C2DL-PSC" :       (1.6 ,1.6),
  "Fluo-N2DH-SIM+" :     (0.125 ,0.125),
  "Fluo-C2DL-Huh7" :     (1,1), ## ??
  }

isbi_by_size = [
  "Fluo-C2DL-Huh7",
  "DIC-C2DH-HeLa",
  "PhC-C2DH-U373",
  "Fluo-N2DH-GOWT1",
  "Fluo-C2DL-MSC",
  "Fluo-C3DL-MDA231",
  "Fluo-N2DH-SIM+",
  "PhC-C2DL-PSC",
  "Fluo-N2DL-HeLa",
  "Fluo-N3DH-CHO",
  "Fluo-C3DH-A549",
  "Fluo-C3DH-A549-SIM",
  "BF-C2DL-MuSC",
  "BF-C2DL-HSC",
  "Fluo-N3DH-SIM+",
  "Fluo-N3DH-CE",
  "Fluo-C3DH-H157",
  "Fluo-N3DL-DRO",
  "Fluo-N3DL-TRIC",
]


isbi_times = {
  "Fluo-C3DH-A549" :     {"01" : (0, 30) ,    "02" : (0, 30)} ,
  "Fluo-C3DH-H157" :     {"01" : (0, 60) ,    "02" : (0, 60)} ,
  "Fluo-C3DL-MDA231" :   {"01" : (0, 12) ,    "02" : (0, 12)} ,
  "Fluo-N3DH-CE" :       {"01" : (0, 195) ,   "02" : (0, 190)} ,
  "Fluo-N3DH-CHO" :      {"01" : (0, 92) ,    "02" : (0, 92)} ,
  "Fluo-N3DL-DRO" :      {"01" : (0, 50) ,    "02" : (0, 50)} ,
  "Fluo-N3DL-TRIC" :     {"01" : (0, 65) ,    "02" : (0, 210)} ,
  "Fluo-C3DH-A549-SIM" : {"01" : (0, 30) ,    "02" : (0, 30)} ,
  "Fluo-N3DH-SIM+"  :    {"01" : (0, 150) ,   "02" : (0, 80)} ,

  "BF-C2DL-HSC" :        {"01" : (0, 1764) ,  "02" : (0, 1764)} ,
  "BF-C2DL-MuSC" :       {"01" : (0, 1376) ,  "02" : (0, 1376)} ,
  "DIC-C2DH-HeLa" :      {"01" : (0, 84) ,    "02" : (0, 84)} ,
  "Fluo-C2DL-MSC" :      {"01" : (0, 48) ,    "02" : (0, 48)} ,
  "Fluo-N2DH-GOWT1" :    {"01" : (0, 92) ,    "02" : (0, 92)} ,
  "Fluo-N2DL-HeLa" :     {"01" : (0, 92) ,    "02" : (0, 92)} ,
  "PhC-C2DH-U373" :      {"01" : (0, 115) ,   "02" : (0, 115)} ,
  "PhC-C2DL-PSC" :       {"01" : (150, 251) , "02" : (150, 251)} ,
  "Fluo-N2DH-SIM+"  :    {"01" : (0, 65) ,    "02" : (0, 150)} ,
  "Fluo-C2DL-Huh7" :     {"01" : (0, 30) ,    "02" : (0, 30)} ,
}