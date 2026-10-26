"""
Step 1 - Download financial news headlines for 2010-01-01 .. 2025-06-30.

Run this on YOUR machine or AWS (it needs internet). Two free sources, pick one
(Guardian is recommended: one consistent source for the whole period, generous limits).

  Guardian Open Platform  - free developer key: https://open-platform.theguardian.com/access/
  NYT Archive API         - free key:          https://developer.nytimes.com/get-started

Usage:
  pip install requests pandas
  python 1_download_news.py --source guardian --key YOUR_KEY
  python 1_download_news.py --source nyt      --key YOUR_KEY

Output: headlines_<source>.csv with columns: published_utc, headline, section, source
The script resumes safely if interrupted (it skips months already saved).
"""
import argparse, os, time, json
import requests
import pandas as pd

START, END = (2010, 1), (2025, 6)


def months():
    y, m = START
    while (y, m) <= END:
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def guardian_month(y, m, key):
    rows, page, pages = [], 1, 1
    last = pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
    while page <= pages:
        r = requests.get("https://content.guardianapis.com/search", params={
            "section": "business", "from-date": f"{y}-{m:02d}-01", "to-date": last.strftime("%Y-%m-%d"),
            "page-size": 200, "page": page, "order-by": "oldest", "api-key": key}, timeout=60)
        r.raise_for_status()
        resp = r.json()["response"]
        pages = resp.get("pages", 1)
        for it in resp.get("results", []):
            rows.append({"published_utc": it["webPublicationDate"], "headline": it["webTitle"],
                         "section": it.get("sectionName", ""), "source": "guardian"})
        page += 1
        time.sleep(0.2)
    return rows


NYT_DESKS = {"Business", "BusinessDay", "Business Day", "SundayBusiness", "Financial", "DealBook"}


def nyt_month(y, m, key):
    r = requests.get(f"https://api.nytimes.com/svc/archive/v1/{y}/{m}.json", params={"api-key": key}, timeout=120)
    r.raise_for_status()
    rows = []
    for d in r.json()["response"]["docs"]:
        sec, desk = d.get("section_name") or "", d.get("news_desk") or ""
        if sec in NYT_DESKS or desk in NYT_DESKS:
            h = (d.get("headline") or {}).get("main") or ""
            if h:
                rows.append({"published_utc": d["pub_date"], "headline": h, "section": sec or desk, "source": "nyt"})
    time.sleep(12.5)  # NYT limit: 5 requests per minute
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["guardian", "nyt"], required=True)
    ap.add_argument("--key", required=True)
    a = ap.parse_args()
    out = f"headlines_{a.source}.csv"
    done_file = out + ".done.json"
    done = set(json.load(open(done_file))) if os.path.exists(done_file) else set()
    fetch = guardian_month if a.source == "guardian" else nyt_month
    for y, m in months():
        tag = f"{y}-{m:02d}"
        if tag in done:
            continue
        for attempt in range(5):
            try:
                rows = fetch(y, m, a.key)
                break
            except Exception as e:
                print(tag, "retry", attempt + 1, e)
                time.sleep(30)
        else:
            raise SystemExit(f"Failed on {tag}; re-run the script to resume.")
        pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        done.add(tag)
        json.dump(sorted(done), open(done_file, "w"))
        print(tag, len(rows), "headlines")
    df = pd.read_csv(out).drop_duplicates(["published_utc", "headline"])
    df.to_csv(out, index=False)
    print("Total headlines:", len(df))


if __name__ == "__main__":
    main()
