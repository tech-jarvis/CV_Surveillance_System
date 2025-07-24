# KPI
- cashier time spent
- cashier missing time
- visitores count
- average time of cashier
- average time of visitors
- Peak Hour

## Data
- Historic Location (Per Person) Using Heatmap
- Visitor Count
  - Customer Count

## Cashier
- Green Shirt
- Sometime black vest

1. Uniform and Appearance Recognition:
Cashiers often wear specific uniforms or have identifiable features such as name tags. You can train a custom classifier to recognize these features.

2. Location-Based Identification: 
Cashiers are typically found at the counter for extended periods. You can use the location within the store and the duration of stay at the counter as key features.

3. Behavioral Patterns: 
Cashiers perform specific tasks such as handling money, scanning items, and interacting frequently with customers. Tracking these behaviors can help in distinguishing them.

# Camera Angle

- 3 = Gate / ATM
- 11 = POS 2 TOP Left
- 13 = Coffie Machine
- 14 = Top Right store, above WC gate
- 15 = WC
- 16 = POS 1 Top Right
- 17 = Top Left store, Coffie Machine
- 18 = POS 1 top
- 23 = POS 2 TOP

# Execution Pipeline

Connect to NAS for accessing videos
* `mount -t nfs -o nfsvers=3 172.16.0.250:/volume1/IDS_KASSA /mnt`

* Step 1: Generate Detection and Tracking Ids, also identify cashier
`train_main_store.py`

Run above code for new video

* Step 1.2: make sure video .csv file is data folder

* Step 2: Run api

`uvicorn main:app --host 0.0.0.0 --port 8009`
