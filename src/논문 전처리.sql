with clean_campaings as (
  select 
    id,
    case when campaign_type = 'trigger' then 1 else 0 end as camp_type_trigger,
    case when campaign_type = 'bulk' then 1 else 0 end as camp_type_bulk,
    case when campaign_type = 'transactional' then 1 else 0  end as camp_type_transactional,
    case when channel = 'email' then 1 else 0 end as camp_channelemail,
    case when channel = 'mobile_push' then 1 else 0 end as camp_channelmobile_push,
    case when channel = 'multichannel' then 1 else 0 end as camp_channelmultichannel,
    case when channel = 'sms' then 1 else 0 end as camp_channelsms,
    case when topic = 'event' then 1 else 0 end as camp_topicevent,
		case when topic = 'happy birthday' then 1 else 0 end as `camp_topichappy_birthday`,
		case when topic = 'leave review' then 1 else 0 end as `camp_topicleave_review`,
		case when topic = 'offer after purchase' then 1 else 0 end as `camp_topicoffer_after_purchase`,
		case when topic = 'sale out' then 1 else 0 end as `camp_topicsale_out`,
		case when topic not in ('event', 'happy birthday', 'leave review', 'offer after purchase', 'sale out') then 1 else 0 end as camp_topicother,
    cast (started_at as timestamp) as started_at,
    cast (finished_at as timestamp) as finished_at
  from `ecommerce-journey.log_data.campaigns`
),

clean_messages as (
  select
    client_id,
    campaign_id,
    cast(sent_at as timestamp) as sent_at,
    cast(purchased_at as timestamp) as purchased_at,
    cast(opened_last_time_at as timestamp) as opened_last_time_at,
    cast(clicked_last_time_at as timestamp) as clicked_last_time_at,
    cast(unsubscribed_at as timestamp) as unsubscribed_at,
    cast(complained_at as timestamp) as complained_at,
    case when channel = 'email' then 1 else 0 end as channel_email,
    case when channel = 'mobile_push' then 1 else 0 end as channel_mobile_push,
    case when channel = 'web_push' then 1 else 0 end as channel_web_push,
    case when channel = 'sms' then 1 else 0 end as channel_sms,
    case when message_type = 'bulk' then 1 else 0 end as message_type_bulk,
    case when message_type = 'transactional' then 1 else 0 end as message_type_transactional,
    case when message_type = 'trigger' then 1 else 0 end as message_type_trigger,
    case when platform = 'desktop' then 1 else 0 end as platform_desktop,
    case when platform = 'smartphone' then 1 else 0 end as platform_smartphone,
    case when platform = 'phablet' then 1 else 0 end as platform_phablet,
    case when platform = 'tablet' then 1 else 0 end as platform_tablet,
    case when platform not in ('desktop', 'smartphone', 'phablet', 'tablet') then 1 else 0 end as platform_,
    case when email_provider in ('gmail.com','gmajl.com') then 1 else 0 end as email_provider_gmail_com,
    case when email_provider = 'mail.ru' then 1 else 0 end as email_provider_mail_ru,
    case when email_provider not in ('gmail.com','gmajl.com', 'mail.ru') or email_provider is null then 1 else 0 end as email_provider_other,
    case when is_opened = true then 1 else 0 end as is_opened,
    case when is_unsubscribed = true then 1 else 0 end as is_unsubscribed,
    case when is_complained = true then 1 else 0 end as is_complained,
    case when is_purchased = true then 1 else 0 end as is_purchased,
    case when is_clicked = true then 1 else 0 end as is_clicked,
    case when is_hard_bounced = true then 1 else 0 end as is_hard_bounced,
    case when is_soft_bounced = true then 1 else 0 end as is_soft_bounced,
    case when is_blocked = true then 1 else 0 end as is_blocked
  from `ecommerce-journey.log_data.messages`
  WHERE sent_at IS NOT NULL
),

clean_client_first_purchase_date as (
  select 
    client_id,
    cast(first_purchase_date as timestamp) as first_purchase_date
  from `ecommerce-journey.log_data.client_first_purchase_date`
),

clean_holidays as (
  select
    date(cast(date as timestamp)) as date,
    1 as is_holiday
  from `ecommerce-journey.log_data.holidays`
),

merge_data as (
  select
    cm.*,
    ccf.* except(client_id),
    cc.* except(id),
    coalesce(ch.is_holiday,0) as is_holiday
  from clean_messages as cm
  left join clean_client_first_purchase_date as ccf on cm.client_id = ccf.client_id
  left join clean_campaings as cc on cm.campaign_id =cc.id
  left join clean_holidays as ch on date(sent_at)=ch.date
),


prep_data AS (
  SELECT 
    *,
    TIMESTAMP_DIFF(finished_at, started_at, SECOND) / 3600.0 AS camp_duration,
    ROW_NUMBER() OVER (PARTITION BY client_id, campaign_id ORDER BY sent_at) AS camp_msg_seq,
    CASE WHEN is_purchased = 1 THEN sent_at END AS ts_purchased,
    CASE WHEN is_opened = 1 THEN sent_at END AS ts_open,
    CASE WHEN is_clicked = 1 THEN sent_at END AS ts_click,
    CASE WHEN is_unsubscribed = 1 THEN sent_at END AS ts_unsub,
    CASE WHEN is_complained = 1 THEN sent_at END AS ts_complaint
  FROM merge_data
),


lagged_features AS (
  SELECT 
    *,
    COUNT(*) OVER (PARTITION BY client_id ORDER BY sent_at ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS total_messages,
    SUM(CASE WHEN camp_msg_seq = 1 THEN 1 ELSE 0 END) 
      OVER (PARTITION BY client_id ORDER BY sent_at ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS total_campaigns,
    COUNT(ts_purchased) OVER (PARTITION BY client_id ORDER BY sent_at ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS total_purchases,
    LAG(camp_duration, 1) OVER (PARTITION BY client_id ORDER BY sent_at) AS prev_camp_duration,
    MIN(first_purchase_date) OVER (PARTITION BY client_id ORDER BY sent_at ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_first_purchase,
    LAST_VALUE(ts_open IGNORE NULLS) OVER (PARTITION BY client_id ORDER BY sent_at ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_open,
    LAST_VALUE(ts_click IGNORE NULLS) OVER (PARTITION BY client_id ORDER BY sent_at ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_click,
    LAST_VALUE(ts_unsub IGNORE NULLS) OVER (PARTITION BY client_id ORDER BY sent_at ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_unsub,
    LAST_VALUE(ts_complaint IGNORE NULLS) OVER (PARTITION BY client_id ORDER BY sent_at ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_complaint

  FROM prep_data
),

raw_diffs AS (
  SELECT
    *,
    TIMESTAMP_DIFF(sent_at, prior_first_purchase, SECOND) / 3600.0 AS diff_first_purchase,
    TIMESTAMP_DIFF(sent_at, prior_open, SECOND) / 3600.0 AS diff_open,
    TIMESTAMP_DIFF(sent_at, prior_click, SECOND) / 3600.0 AS diff_click,
    TIMESTAMP_DIFF(sent_at, prior_unsub, SECOND) / 3600.0 AS diff_unsub,
    TIMESTAMP_DIFF(sent_at, prior_complaint, SECOND) / 3600.0 AS diff_complaint
  FROM lagged_features
),

medians AS (
  SELECT
    PERCENTILE_CONT(diff_first_purchase, 0.5) OVER() AS med_first_purchase,
    PERCENTILE_CONT(diff_open, 0.5) OVER() AS med_open,
    PERCENTILE_CONT(diff_click, 0.5) OVER() AS med_click,
    PERCENTILE_CONT(diff_unsub, 0.5) OVER() AS med_unsub,
    PERCENTILE_CONT(diff_complaint, 0.5) OVER() AS med_complaint,
    PERCENTILE_CONT(prev_camp_duration, 0.5) OVER() AS med_camp_duration
  FROM raw_diffs
  LIMIT 1
)

SELECT 
  d.* EXCEPT(
      camp_msg_seq, camp_duration,
      ts_purchased, ts_open, ts_click, ts_unsub, ts_complaint,
      prior_first_purchase, prior_open, prior_click, prior_unsub, prior_complaint,
      diff_first_purchase, diff_open, diff_click, diff_unsub, diff_complaint,
      prev_camp_duration
  ),
  
  COALESCE(d.total_messages, 0) AS total_messages,
  COALESCE(d.total_campaigns, 0) AS total_campaigns,
  COALESCE(d.total_purchases, 0) AS total_purchases,

  GREATEST(COALESCE(d.prev_camp_duration, m.med_camp_duration), 0) AS avg_campaign_duration,
  GREATEST(COALESCE(d.diff_first_purchase, m.med_first_purchase), 0) AS avg_time_since_first_purchase,
  GREATEST(COALESCE(d.diff_open, m.med_open), 0) AS avg_time_since_last_open,
  GREATEST(COALESCE(d.diff_click, m.med_click), 0) AS avg_time_since_last_click,
  GREATEST(COALESCE(d.diff_unsub, m.med_unsub), 0) AS avg_time_since_unsubscribe,
  GREATEST(COALESCE(d.diff_complaint, m.med_complaint), 0) AS avg_time_since_complaint

FROM raw_diffs d
CROSS JOIN medians m