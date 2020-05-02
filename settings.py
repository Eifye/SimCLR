
# num of input channnels
inch = 3 # color

# resnet blocks
blocks = [3, 4, 6, 3] # res50

# num of intermediate outputs of resnet
num_outputs = len(blocks)

# num of feature dims for contrastive learning
proj_dim = 128

epochs = 100000

lr = 0.01

t_max = 500
lr_min = 0.0001

batch_size = 16

crop_size = (128, 128) # (h,w)

strength = 0.3 # data aug strength [0..0.5]

#data_dirs = ['D:/Dev/old/datas/danbooru2018/512px/0000',]

data_dirs = ["D:/Dev/old/datas/danbooru2018/512px/0000",
"D:/Dev/old/datas/danbooru2018/512px/0001",
"D:/Dev/old/datas/danbooru2018/512px/0002",
"D:/Dev/old/datas/danbooru2018/512px/0003",
"D:/Dev/old/datas/danbooru2018/512px/0004",
"D:/Dev/old/datas/danbooru2018/512px/0005",
"D:/Dev/old/datas/danbooru2018/512px/0006",
"D:/Dev/old/datas/danbooru2018/512px/0007",
"D:/Dev/old/datas/danbooru2018/512px/0008",
"D:/Dev/old/datas/danbooru2018/512px/0009",
"D:/Dev/old/datas/danbooru2018/512px/0010",
"D:/Dev/old/datas/danbooru2018/512px/0011",
"D:/Dev/old/datas/danbooru2018/512px/0012",
"D:/Dev/old/datas/danbooru2018/512px/0013",
"D:/Dev/old/datas/danbooru2018/512px/0014",
"D:/Dev/old/datas/danbooru2018/512px/0015",
"D:/Dev/old/datas/danbooru2018/512px/0016",
"D:/Dev/old/datas/danbooru2018/512px/0017",
"D:/Dev/old/datas/danbooru2018/512px/0018",
"D:/Dev/old/datas/danbooru2018/512px/0019",
"D:/Dev/old/datas/danbooru2018/512px/0020",
"D:/Dev/old/datas/danbooru2018/512px/0021",
"D:/Dev/old/datas/danbooru2018/512px/0022",
"D:/Dev/old/datas/danbooru2018/512px/0023",
"D:/Dev/old/datas/danbooru2018/512px/0024",
"D:/Dev/old/datas/danbooru2018/512px/0025",
"D:/Dev/old/datas/danbooru2018/512px/0026",
"D:/Dev/old/datas/danbooru2018/512px/0027",
"D:/Dev/old/datas/danbooru2018/512px/0028",
"D:/Dev/old/datas/danbooru2018/512px/0029",
"D:/Dev/old/datas/danbooru2018/512px/0030",
"D:/Dev/old/datas/danbooru2018/512px/0031",
"D:/Dev/old/datas/danbooru2018/512px/0032",
"D:/Dev/old/datas/danbooru2018/512px/0033",
"D:/Dev/old/datas/danbooru2018/512px/0034",
"D:/Dev/old/datas/danbooru2018/512px/0035",
"D:/Dev/old/datas/danbooru2018/512px/0036",
"D:/Dev/old/datas/danbooru2018/512px/0037",
"D:/Dev/old/datas/danbooru2018/512px/0038",
"D:/Dev/old/datas/danbooru2018/512px/0039",
"D:/Dev/old/datas/danbooru2018/512px/0040",
"D:/Dev/old/datas/danbooru2018/512px/0041",
"D:/Dev/old/datas/danbooru2018/512px/0042",
"D:/Dev/old/datas/danbooru2018/512px/0043",
"D:/Dev/old/datas/danbooru2018/512px/0044",
"D:/Dev/old/datas/danbooru2018/512px/0045",
"D:/Dev/old/datas/danbooru2018/512px/0046",
"D:/Dev/old/datas/danbooru2018/512px/0047",
"D:/Dev/old/datas/danbooru2018/512px/0048",
"D:/Dev/old/datas/danbooru2018/512px/0049",
"D:/Dev/old/datas/danbooru2018/512px/0050",
"D:/Dev/old/datas/danbooru2018/512px/0051",
"D:/Dev/old/datas/danbooru2018/512px/0052",
"D:/Dev/old/datas/danbooru2018/512px/0053",
"D:/Dev/old/datas/danbooru2018/512px/0054",
"D:/Dev/old/datas/danbooru2018/512px/0055",
"D:/Dev/old/datas/danbooru2018/512px/0056",
"D:/Dev/old/datas/danbooru2018/512px/0057",
"D:/Dev/old/datas/danbooru2018/512px/0058",
"D:/Dev/old/datas/danbooru2018/512px/0059",
"D:/Dev/old/datas/danbooru2018/512px/0060",
"D:/Dev/old/datas/danbooru2018/512px/0061",
"D:/Dev/old/datas/danbooru2018/512px/0062",
"D:/Dev/old/datas/danbooru2018/512px/0063",
"D:/Dev/old/datas/danbooru2018/512px/0064",
"D:/Dev/old/datas/danbooru2018/512px/0065",
"D:/Dev/old/datas/danbooru2018/512px/0066",
"D:/Dev/old/datas/danbooru2018/512px/0067",
"D:/Dev/old/datas/danbooru2018/512px/0068",
"D:/Dev/old/datas/danbooru2018/512px/0069",
"D:/Dev/old/datas/danbooru2018/512px/0070",
"D:/Dev/old/datas/danbooru2018/512px/0071",
"D:/Dev/old/datas/danbooru2018/512px/0072",
"D:/Dev/old/datas/danbooru2018/512px/0073",
"D:/Dev/old/datas/danbooru2018/512px/0074",
"D:/Dev/old/datas/danbooru2018/512px/0075",
"D:/Dev/old/datas/danbooru2018/512px/0076",
"D:/Dev/old/datas/danbooru2018/512px/0077",
"D:/Dev/old/datas/danbooru2018/512px/0078",
"D:/Dev/old/datas/danbooru2018/512px/0079",
"D:/Dev/old/datas/danbooru2018/512px/0080",
"D:/Dev/old/datas/danbooru2018/512px/0081",
"D:/Dev/old/datas/danbooru2018/512px/0082",
"D:/Dev/old/datas/danbooru2018/512px/0083",
"D:/Dev/old/datas/danbooru2018/512px/0084",
"D:/Dev/old/datas/danbooru2018/512px/0085",
"D:/Dev/old/datas/danbooru2018/512px/0086",
"D:/Dev/old/datas/danbooru2018/512px/0087",
"D:/Dev/old/datas/danbooru2018/512px/0088",
"D:/Dev/old/datas/danbooru2018/512px/0089",
"D:/Dev/old/datas/danbooru2018/512px/0090",
"D:/Dev/old/datas/danbooru2018/512px/0091",
"D:/Dev/old/datas/danbooru2018/512px/0092",
"D:/Dev/old/datas/danbooru2018/512px/0093",
"D:/Dev/old/datas/danbooru2018/512px/0094",
"D:/Dev/old/datas/danbooru2018/512px/0095",
"D:/Dev/old/datas/danbooru2018/512px/0096",
"D:/Dev/old/datas/danbooru2018/512px/0097",
"D:/Dev/old/datas/danbooru2018/512px/0098",
"D:/Dev/old/datas/danbooru2018/512px/0099",
"D:/Dev/old/datas/danbooru2018/512px/0100",
"D:/Dev/old/datas/danbooru2018/GAME_DUMP/FGO",
"D:/Dev/old/datas/danbooru2018/GAME_DUMP/GBF",]
