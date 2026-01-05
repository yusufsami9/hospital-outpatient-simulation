# Hospital Outpatient Clinic Simulation (SimPy)

This repository contains a **discrete-event simulation** of an orthopedic outpatient department,
implemented in **Python using SimPy**.

The model simulates patient flow through **doctor examinations and X-ray services**, incorporating
appointments, walk-ins, lunch breaks, priority rules, and resource constraints to analyze
**queue lengths, waiting times, and resource utilization**.

---

## Problem Description

The outpatient clinic operates with:
- Multiple doctors
- Limited X-ray rooms
- A mix of **appointment-based** and **walk-in** patients
- Time-dependent behaviors such as lunch breaks and afternoon speed-up

Patients may require:
1. First examination
2. X-ray imaging
3. Second examination (if X-ray is required)

The objective is to analyze how scheduling policies and resource allocation affect queues,
waiting times, and throughput.

---

## Modeling Approach

- **Discrete-event simulation** using SimPy
- Priority-based resource allocation
- Doctor-specific stochastic service-time distributions
- Appointment punctuality modeled via uniform deviation
- Time-dependent service speed (afternoon speed-up)
- Slower X-ray service before lunch end

---

## Key Features

- Appointment and walk-in patient streams
- Priority handling for appointments vs. walk-ins
- Doctor lunch breaks with service suspension
- Second examination routing after X-ray
- Multiple X-ray rooms with queue balancing
- Detailed event logging
- Queue length tracking over time
- Performance statistics and visualizations

---

## Supported Patient Types

- Appointment patients
- Walk-in patients
- Type A: requires X-ray
- Type B: no X-ray

---

## Simulation Policies

- **Priority rule**:
  - Appointments are prioritized using their scheduled time
  - Walk-ins receive a large priority offset to ensure lower priority
- **Special doctor roles** (configurable):
  - Appointment-only doctor
  - Walk-in-only doctor
- **Lunch break**:
  - Doctors pause service between predefined times
- **Afternoon speed-up**:
  - Doctor service times are reduced after lunch

---

## Repository Structure
```
hospital-simulation/
├── main.py
├── outputs/
│ ├── doctor_queue_lengths.png
│ └── xray_queue_lengths.png
├── README.md
```

---



---

## Requirements

- Python 3.x
- simpy
- numpy
- matplotlib

Install dependencies with:
```bash
pip install simpy numpy matplotlib
```
## How to Run

Run the simulation with:
python main.py

Simulation parameters (e.g., number of doctors, lunch times, priority rules)
are defined at the top of main.py.

## Output

The simulation produces:

- Detailed event logs (arrival, service start/end, departure)

- Queue length plots for:

    - Doctors
    - X-ray rooms

- Summary statistics including:

  - Total patients served

  - Appointment vs. walk-in counts

  - Doctor workloads

  - X-ray utilization

  - Referral rates per doctor




