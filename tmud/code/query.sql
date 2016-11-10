

SELECT event_id,
       app_id,
       is_installed,
       is_active,
       app_label_cate.label_id,
        app_label_cate.category
FROM app_events
  LEFT OUTER JOIN (
  SELECT app_id as app_id2,
         label_id,
         label_ca.category category
  FROM app_labels
    LEFT OUTER JOIN
      (SELECT label_id as label_id2,
              category
      FROM label_categories) label_ca ON app_labels.label_id = label_ca.label_id2) app_label_cate
  ON app_events.app_id = app_label_cate.app_id2



  -- 每个手机game类型app的安装数量
  select
    device_id,
    num,
    CASE category
      WHEN 'Games' THEN num
      ELSE 0
    END AS GAMES,
    CASE category
      WHEN 'Property' THEN num
      ELSE 0
    END AS Property,
    CASE category
      WHEN 'Family' THEN num
      ELSE 0
    END AS Family,
    CASE category
      WHEN 'Fun' THEN num
      ELSE 0
    END AS Fun,
    CASE category
      WHEN 'Productivity' THEN num
      ELSE 0
    END AS Productivity,
    CASE category
      WHEN 'Finance' THEN num
      ELSE 0
    END AS Finance,
    CASE category
      WHEN 'Services' THEN num
      ELSE 0
    END AS Services,
    CASE category
      WHEN 'Travel' THEN num
      ELSE 0
    END AS Travel,
    CASE category
      WHEN 'Video' THEN num
      ELSE 0
    END AS Video,
    CASE category
      WHEN 'Shopping' THEN num
      ELSE 0
    END AS Shopping,
    CASE category
      WHEN 'Education' THEN num
      ELSE 0
    END AS Education,
    CASE category
      WHEN 'Sports' THEN num
      ELSE 0
    END AS Sports,
    CASE category
      WHEN 'Music' THEN num
      ELSE 0
    END AS Music
  from
  (select device_id,
         category,
         count(distinct app_id) num
  from tmud_main group by device_id, category) A

  -- 特征f2 玩游戏 购物 看视频 音乐 运动 体育的时间
  select A.device_id as device_id,
         A.category category,
         percentile_approx(hour(A.ts),0.25)  as active_25_time,
         percentile_approx(hour(A.ts),0.5)  as active_50_time,
         percentile_approx(hour(A.ts),0.75)  as active_75_time
  from
  (select device_id,
         category,
         ts
  from tmud_main  group by device_id, event_id, ts,category  ) A
  group by A.device_id, A.category



