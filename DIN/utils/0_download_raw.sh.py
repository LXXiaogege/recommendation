#! /bin/bash
# 下载Amazon Electronic数据
# cd ../raw_data
# wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz
# gzip -d reviews_Electronics_5.json.gz
# wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz
# gzip -d meta_Electronics.json.gz


"""
Electronics_5.json :  reviews（评论）

字段:
reviewerID： 评论用户id
asin： 商品id
reviewerName：评论用户名字
helpful：该评论帮助度评分 helpfulness rating of the review, e.g. 2/3
reviewText：评论内容
overall：评分
summary：评价总结
unixReviewTime:评价时间戳
reviewTime：评价时间
例子：
{
  "reviewerID": "A2SUAM1J3GNN3B",
  "asin": "0000013714",
  "reviewerName": "J. McDonald",
  "helpful": [2, 3],
  "reviewText": "I bought this for my husband who plays the piano.  He is having a wonderful time playing these old hymns.  The music  is at times hard to read because we think the book was published for singing from more than playing from.  Great purchase though!",
  "overall": 5.0,
  "summary": "Heavenly Highway Hymns",
  "unixReviewTime": 1252800000,
  "reviewTime": "09 13, 2009"
}



meta_Electronics.json: product metadata(商品元数据)

字段：
asin：产品id
title：商品名
price：价格（美元）
imURL：产品图片URL
related：相关产品
brand：品牌
categories：商品类别
例子：
{
  "asin": "0000031852",
  "title": "Girls Ballet Tutu Zebra Hot Pink",
  "price": 3.17,
  "imUrl": "http://ecx.images-amazon.com/images/I/51fAmVkTbyL._SY300_.jpg",
  "related":
  {
    "also_bought": ["B00JHONN1S", "B002BZX8Z6"],
    "also_viewed": ["B002BZX8Z6", "B00JHONN1S"],
    "bought_together": ["B002BZX8Z6"]
  },
  "salesRank": {"Toys & Games": 211836},
  "brand": "Coxlures",
  "categories": [["Sports & Outdoors", "Other Sports", "Dance"]]
}

"""
