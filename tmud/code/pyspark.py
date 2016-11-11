# 设置pyspark不断出现的log信息
http://stackoverflow.com/questions/25193488/how-to-turn-off-info-logging-in-pyspark
def quiet_logs( sc ):
  logger = sc._jvm.org.apache.log4j
  logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
  logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )


app_labels_rdd = sc.textFile("hdfs://10-149-11-153:9000/user/guoqiang/tmud/app_labels.csv")
label_categories_rdd = sc.textFile("hdfs://10-149-11-153:9000/user/guoqiang/tmud/label_categories.csv")
label_categories_rdd.persist()
label_categories_rdd.take(1)
[u'label_id,category']

label_categories_rdd = label_categories_rdd.map(lambda line:line.split(","))

label_categories_rdd = label_categories_rdd.filter(lambda x: x[0]!='label_id')

label_categories_rdd.count()

label_categories_rdd = label_categories_rdd.map(lambda x:[x[0],to_Games(x[1])])
label_categories_rdd = label_categories_rdd.map(lambda x:[x[0],to_Property(x[1])])
label_categories_rdd = label_categories_rdd.map(lambda x:[x[0],to_Family(x[1])])
label_categories_rdd = label_categories_rdd.map(lambda x:[x[0],to_Fun(x[1])])
label_categories_rdd = label_categories_rdd.map(lambda x:[x[0],to_Productivity(x[1])])
label_categories_rdd = label_categories_rdd.map(lambda x:[x[0],to_Finance(x[1])])
label_categories_rdd = label_categories_rdd.map(lambda x:[x[0],to_Religion(x[1])])
label_categories_rdd = label_categories_rdd.map(lambda x:[x[0],to_Services(x[1])])
label_categories_rdd = label_categories_rdd.map(lambda x:[x[0],to_Travel(x[1])])
label_categories_rdd = label_categories_rdd.map(lambda x:[x[0],to_Custom(x[1])])
label_categories_rdd = label_categories_rdd.map(lambda x:[x[0],to_Video(x[1])])
label_categories_rdd = label_categories_rdd.map(lambda x:[x[0],to_Shopping(x[1])])
label_categories_rdd = label_categories_rdd.map(lambda x:[x[0],to_Education(x[1])])
label_categories_rdd = label_categories_rdd.map(lambda x:[x[0],to_Vitality(x[1])])
label_categories_rdd = label_categories_rdd.map(lambda x:[x[0],to_Sports(x[1])])
label_categories_rdd = label_categories_rdd.map(lambda x:[x[0],to_Music(x[1])])
label_categories_rdd = label_categories_rdd.map(lambda x:[x[0],to_Travel_2(x[1])])
label_categories_rdd = label_categories_rdd.map(lambda x:[x[0],to_Other(x[1])])

label_categories_df = label_categories_rdd.toDF(['label_id','category'])
label_categories_df.persist()


# app_labels_rdd->app_labels_df
[u'app_id,label_id']

app_labels_rdd = app_labels_rdd.map(lambda line:line.split(","))
app_labels_rdd = app_labels_rdd.filter(lambda x: x[0]!='app_id')
app_labels_rdd.take(1)
app_labels_df = app_labels_rdd.toDF(['app_id','label_id'])
app_labels_df.show()

# events_rdd -> events_df
[u'event_id,device_id,timestamp,longitude,latitude']

events_rdd = sc.textFile("hdfs://10-149-11-153:9000/user/guoqiang/tmud/events.csv")
events_rdd.persist()
events_rdd.take(1)
events_rdd = events_rdd.map(lambda line:line.split(","))
events_rdd = events_rdd.filter(lambda x: x[0]!='event_id')
events_rdd.persist()
events_df = events_rdd.toDF(['event_id','device_id','timestamp','longitude','latitude'])
events_df.show()


# phone_brand_device_model.csv
[u'device_id,phone_brand,device_model']
phone_brand_device_model_rdd = sc.textFile("hdfs://10-149-11-153:9000/user/guoqiang/tmud/phone_brand_device_model.csv")
phone_brand_device_model_rdd.persist()
phone_brand_device_model_rdd = phone_brand_device_model_rdd.map(lambda line:line.split(","))
phone_brand_device_model_rdd = phone_brand_device_model_rdd.filter(lambda x: x[0]!='device_id')
phone_brand_device_model_rdd.take(1)
phone_brand_device_model_df = phone_brand_device_model_rdd.toDF(['device_id','phone_brand','device_model'])
phone_brand_device_model_df.show()

#wirte as parquet
app_events_df.write.parquet("hdfs://10-149-11-153:9000/user/guoqiang/tmud/app_events.parquet")
label_categories_df.write.parquet("hdfs://10-149-11-153:9000/user/guoqiang/tmud/label_categories.parquet")
app_labels_df.write.parquet("hdfs://10-149-11-153:9000/user/guoqiang/tmud/app_labels.parquet")
events_df.write.parquet("hdfs://10-149-11-153:9000/user/guoqiang/tmud/events.parquet")
phone_brand_device_model_df.write.parquet("hdfs://10-149-11-153:9000/user/guoqiang/tmud/phone_brand_device_model.parquet")

# register table for dataframe
app_events_df.registerTempTable("app_events")
label_categories_df.registerTempTable("label_categories")
app_labels_df.registerTempTable("app_labels")
events_df.registerTempTable("events")
phone_brand_device_model_df.registerTempTable("phone_brand_device_model")
# sql
"app_events":
['event_id,app_id,is_installed,is_active']

result = sqlContext.sql("SELECT * FROM app_events LEFT OUTER JOIN (SELECT app_id as app_id2,label_id as label_id1 FROM app_labels app_la LEFT OUTER JOIN (SELECT label_id as label_id2, category FROM label_categories) label_ca ON app_la.label_id1 = label_ca.label_id2) app_label_cate ON app_events.app_id = app_label_cate.app_id2")
result = sqlContext.sql("  SELECT app_id,label_id,label_ca.category category FROM app_labels LEFT OUTER JOIN (SELECT label_id as label_id2,category FROM label_categories) label_ca ON app_labels.label_id = label_ca.label_id2")
result = sqlContext.sql("SELECT event_id,"
                        "app_id,"
                        "is_installed,"
       "is_active,"
       "app_label_cate.label_id,"
        "app_label_cate.category"
  " FROM app_events"
  " LEFT OUTER JOIN ("
  "SELECT app_id as app_id2,"
         "label_id,"
         "label_ca.category category"
  " FROM app_labels"
    " LEFT OUTER JOIN"
      "(SELECT label_id as label_id2,"
              "category"
      " FROM label_categories) label_ca ON app_labels.label_id = label_ca.label_id2) app_label_cate"
  " ON app_events.app_id = app_label_cate.app_id2")

# app_main代表了app细节的所有数据,没有对数据进行相关字段的细化处理
result.write.parquet("hdfs://10-149-11-153:9000/user/guoqiang/tmud/app_main_table.parquet")
result.registerTempTable("app_main")
#|event_id|app_id|is_installed|is_active|label_id| category|
app_main_df = result


#71376435
app_main_df.filter(app_main_df['is_active']>0).count()
#12732996
app_events_df.filter(app_events_df['is_active']>0).count()

result = sqlContext.sql("select is_active, count(distinct event_id) distinct_event_num from app_main group by is_active")
+---------+------------------+
|is_active|distinct_event_num|
+---------+------------------+
|        0|            625168|
|        1|           1477059|
+---------+------------------+
result.show()

### 进行相关去重
result = sqlContext.sql("select event_id,app_id,max(is_active) is_active,max(category) category from app_main group by event_id,app_id")
result.count()
result.show()
result.registerTempTable("app_main_small")
+--------+--------------------+---------+--------+
|event_id|              app_id|is_active|category|
+--------+--------------------+---------+--------+

### 整合总表,用于后续的所有数据计算和统计

result = sqlContext.sql("select * from events limit 100")


final_result = sqlContext.sql("select events.event_id event_id,"
                        "events.device_id,"
                        "events.timestamp ts,"
                        "events.longitude,"
                        "events.latitude, "
                        "small.app_id app_id, "
                        "small.is_active is_active,"
                        " small.category category,"
                        "phone.phone_brand phone_brand, "
                        "phone.device_model device_model"
                        " from events left outer join"
                        " (select event_id event_id2, app_id, is_active, category from app_main_small) small on events.event_id=small.event_id2"
                        " left outer join (select device_id device_id2,phone_brand,device_model from phone_brand_device_model) phone on events.device_id=phone.device_id2")
result = sqlContext.sql("select device_id, case category when 'Games' then 1 else 0 end as game from tmud_main")

final_reuslt.registerTempTable("tmud_main")
final_reuslt.write.parquet("hdfs://10-149-11-153:9000/user/guoqiang/tmud/tmud_main_table.parquet")
df = sqlContext.read.load("hdfs://10-149-11-153:9000/user/guoqiang/tmud/tmud_main_table.parquet")
df.registerTempTable("tmud_main")
#final_result schema

+--------+--------------------+-------------------+---------+--------+--------------------+---------+--------+-----------+------------+
|event_id|           device_id|                 ts|longitude|latitude|              app_id|is_active|category|phone_brand|device_model|
+--------+--------------------+-------------------+---------+--------+--------------------+---------+--------+-----------+------------+
| 1376906|-1005647052671742462|2016-05-05 02:00:48|     0.00|    0.00|-1633873313139722876|        0|  Travel|       OPPO|       U705T|
| 1376906|-1005647052671742462|2016-05-05 02:00:48|     0.00|    0.00| 7536856914263861653|        0|Services|       OPPO|       U705T|
+--------+--------------------+-------------------+---------+--------+--------------------+---------+--------+-----------+------------+


result = sqlContext.sql("select device_id, "
                        "count(distinct app_id) app_num, "
                        "count(distinct event_id) event_num, "
                        "count(distinct category) cate_num, "
                        "count(distinct device_model) model_num "
                        " from tmud_main group by device_id sort by app_num desc")

result.registerTempTable("f1")



result = sqlContext.sql("select from_unixtime(unix_timestamp(ts, 'yyyy-MM-dd HH:mm:ss')) from tmud_main limit 100")

result_active_time = sqlContext.sql("select A.device_id as device_id4, "
                        "percentile_approx(hour(A.ts),0.5)  as all_active_50_time, "
                        "percentile_approx(hour(A.ts),0.25)  as all_active_25_time, "
                        "percentile_approx(hour(A.ts),0.75)  as allactive_75_time"
                        "  from (select device_id,ts from tmud_main group by device_id,ts) A "
                        "group by A.device_id")
result_active_time.registerTempTable("f4")


result = sqlContext.sql("  select\
                            device_id,\
                            num,\
                            CASE category\
                              WHEN 'Games' THEN 1\
                              ELSE 0\
                            END AS GAMES\
                          from\
                          (select device_id,\
                                 category,\
                                 count(distinct app_id) num\
                          from tmud_main group by device_id, category) A order by GAMES desc")

result = sqlContext.sql("  select A.device_id as Games_device_id,\
         percentile_approx(hour(A.ts),0.25)  as Games_active_25_time,\
         percentile_approx(hour(A.ts),0.5)  as Games_active_50_time,\
         percentile_approx(hour(A.ts),0.75)  as Games_active_75_time\
  from \
  (select device_id,\
         ts\
  from tmud_main where category='Games'  group by device_id, event_id, ts  ) A \
  group by A.device_id\
")
result.registerTempTable("f_Games")

result = sqlContext.sql("  select A.device_id as Video_device_id,\
         percentile_approx(hour(A.ts),0.25)  as Video_active_25_time,\
         percentile_approx(hour(A.ts),0.5)  as Video_active_50_time,\
         percentile_approx(hour(A.ts),0.75)  as Video_active_75_time\
  from \
  (select device_id,\
         ts\
  from tmud_main where category='Video'  group by device_id, event_id, ts  ) A \
  group by A.device_id\
")
result.registerTempTable("f_Video")

result = sqlContext.sql("  select A.device_id as Shopping_device_id,\
         percentile_approx(hour(A.ts),0.25)  as Shopping_active_25_time,\
         percentile_approx(hour(A.ts),0.5)  as Shopping_active_50_time,\
         percentile_approx(hour(A.ts),0.75)  as Shopping_active_75_time\
  from \
  (select device_id,\
         ts\
  from tmud_main where category='Shopping'  group by device_id, event_id, ts  ) A \
  group by A.device_id\
")
result.registerTempTable("f_Shopping")

result = sqlContext.sql("  select A.device_id as Education_device_id,\
         percentile_approx(hour(A.ts),0.25)  as Education_active_25_time,\
         percentile_approx(hour(A.ts),0.5)  as Education_active_50_time,\
         percentile_approx(hour(A.ts),0.75)  as Education_active_75_time\
  from \
  (select device_id,\
         ts\
  from tmud_main where category='Education'  group by device_id, event_id, ts  ) A \
  group by A.device_id\
")
result.registerTempTable("f_Education")

result = sqlContext.sql("  select A.device_id as Finance_device_id,\
         percentile_approx(hour(A.ts),0.25)  as Finance_active_25_time,\
         percentile_approx(hour(A.ts),0.5)  as Finance_active_50_time,\
         percentile_approx(hour(A.ts),0.75)  as Finance_active_75_time\
  from \
  (select device_id,\
         ts\
  from tmud_main where category='Finance'  group by device_id, event_id, ts  ) A \
  group by A.device_id\
")
result.registerTempTable("f_Finance")

result = sqlContext.sql("  select A.device_id as Sports_device_id,\
         percentile_approx(hour(A.ts),0.25)  as Sports_active_25_time,\
         percentile_approx(hour(A.ts),0.5)  as Sports_active_50_time,\
         percentile_approx(hour(A.ts),0.75)  as Sports_active_75_time\
  from \
  (select device_id,\
         ts\
  from tmud_main where category='Sports'  group by device_id, event_id, ts  ) A \
  group by A.device_id\
")
result.registerTempTable("f_Sports")


result = sqlContext.sql("  select A.device_id as Music_device_id,\
         percentile_approx(hour(A.ts),0.25)  as Music_active_25_time,\
         percentile_approx(hour(A.ts),0.5)  as Music_active_50_time,\
         percentile_approx(hour(A.ts),0.75)  as Music_active_75_time\
  from \
  (select device_id,\
         ts\
  from tmud_main where category='Music'  group by device_id, event_id, ts  ) A \
  group by A.device_id\
")
result.registerTempTable("f_Music")




result = sqlContext.sql("select\
    device_id as device_id3,\
    CASE category\
      WHEN 'Games' THEN num\
      ELSE 0\
    END AS GAMES,\
    CASE category\
      WHEN 'Property' THEN num\
      ELSE 0\
    END AS Property,\
    CASE category\
      WHEN 'Family' THEN num\
      ELSE 0\
    END AS Family,\
    CASE category\
      WHEN 'Fun' THEN num\
      ELSE 0\
    END AS Fun,\
    CASE category\
      WHEN 'Productivity' THEN num\
      ELSE 0\
    END AS Productivity,\
    CASE category\
      WHEN 'Finance' THEN num\
      ELSE 0\
    END AS Finance,\
    CASE category\
      WHEN 'Services' THEN num\
      ELSE 0\
    END AS Services,\
    CASE category\
      WHEN 'Travel' THEN num\
      ELSE 0\
    END AS Travel,\
    CASE category\
      WHEN 'Video' THEN num\
      ELSE 0\
    END AS Video,\
    CASE category\
      WHEN 'Shopping' THEN num\
      ELSE 0\
    END AS Shopping,\
    CASE category\
      WHEN 'Education' THEN num\
      ELSE 0\
    END AS Education,\
    CASE category\
      WHEN 'Sports' THEN num\
      ELSE 0\
    END AS Sports,\
    CASE category\
      WHEN 'Music' THEN num\
      ELSE 0\
    END AS Music\
  from\
  (select device_id,\
         category,\
         count(distinct app_id) num\
  from tmud_main group by device_id, category) A")

result.registerTempTable("f3")


result.registerTempTable("tmud_main")
# 总的数据量是60865
result.write.format("json").save('hdfs://10-149-11-153:9000/user/guoqiang/tmud/tmud_main0806.json')
#将hdfs的文件导成单个文件下载到本地
hadoop fs -getmerge /user/guoqiang/tmud/tmud_main0806.json ./tmud_main_0806.json

['Education','Finance', 'Sports', 'Music']
result = sqlContext.sql("select * "
                        "from f1 left outer join f_Games on f1.device_id==f_Games.Games_device_id "
                        "left outer join f_Video on f1.device_id==f_Video.Video_device_id "
                        "left outer join f_Shopping on f1.device_id==f_Shopping.Shopping_device_id "
                        "left outer join f_Education on f1.device_id==f_Education.Education_device_id "
                        "left outer join f_Finance on f1.device_id==f_Finance.Finance_device_id "
                        "left outer join f_Sports on f1.device_id==f_Sports.Sports_device_id "
                        "left outer join f_Music on f1.device_id==f_Music.Music_device_id "
                        "left outer join f3 on f1.device_id == f3.device_id3 "
                        "left outer join f4 on f1.device_id == f4.device_id4")

result = result.drop('Games_device_id')
result = result.drop('Video_device_id')
result = result.drop('Shopping_device_id')
result = result.drop('Education_device_id')
result = result.drop('Finance_device_id')
result = result.drop('Sports_device_id')
result = result.drop('Music_device_id')
result = result.drop('device_id3')
result = result.drop('device_id4')

result.printSchema()

sqlContext.sql("select * from f1").show()

result = sqlContext.sql("select * from f1 left outer join f2 on f1.device_id==f2.device_id "
                        ).show()