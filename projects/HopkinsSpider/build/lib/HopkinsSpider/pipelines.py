# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import json
import os
class HopkinsspiderPipeline:
    def __init__(self):
        if(os.path.exists('./crawler_results') == False):
            os.makedirs('./crawler_results')
        self.file = open('./crawler_results/hopkins.json','w',encoding='utf-8')
    def process_item(self, item, spider):
        self.file.write(json.dumps(dict(item),ensure_ascii=False) + "\n")
        return item
    def __del__(self):
        self.file.close()