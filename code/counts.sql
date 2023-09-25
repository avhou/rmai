install json; 
load json;
create table data as select * from read_json_auto('data/annotations/instances_train2017.json', maximum_object_size=1000000000);
copy (
  select a.category_id, 
  count(distinct a.image_id) as distinct_count, 
  count(a.image_id) as total_count 
  from (select unnest(annotations) as a from data) 
  where a.category_id in (12, 13, 19, 24, 41, 42, 48, 49, 50, 78, 79) 
  group by a.category_id order by 1
) to 'counts.csv' (header, delimiter ',');
