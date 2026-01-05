# -*- coding: utf-8 -*-

import simpy
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import sys
import datetime

sys.setrecursionlimit(2000)

# ----------- Configuration -----------
SIM_TIME = 480
WALKIN_CUTOFF_TIME = SIM_TIME - 60
LUNCH_START = 240
LUNCH_END = 300
NUM_DOCTORS = 7
NUM_XRAY_ROOMS = 2
RANDOM_SEED = 42
QUEUE_TRACK_INTERVAL = 1
AFTERNOON_SPEEDUP_FACTOR = 0.85

# --- Uniform distribution parameters for appointment punctuality ---
UNIFORM_MIN_DEVIATION_MINUTES = -5
UNIFORM_MAX_DEVIATION_MINUTES = 10

# Large offset to separate walk-in priority from appointment priority
WALKIN_PRIORITY_OFFSET = SIM_TIME * 1000

# --- Special doctor roles (0-based IDs) ---
APPOINTMENT_ONLY_DOCTOR_ID = 0
WALKIN_ONLY_DOCTOR_ID = 1

random.seed(RANDOM_SEED)

xray_resources = []
doctors = []

# Doctor configurations
doctor_configs = [
    {  # Doctor 1 (ID 0)
        "arrival": lambda: random.expovariate(1 / 10),
        "type_a_first_exam": lambda: max(random.gammavariate(1.15, 2.41), 0.25),
        "type_b_first_exam": lambda: max(random.gammavariate(0.80, 4.71), 0.70),
        "type_a_second_exam": lambda: max(random.gammavariate(2.3, 1.5), 0.25),
        "appointment_interval": lambda: max(random.uniform(9, 15), 10),
        "xray_probability": 0.76
    },
    {  # Doctor 2 (ID 1)
        "arrival": lambda: random.expovariate(1 / 12),
        "type_a_first_exam": lambda: max(random.gammavariate(3, 1), 0.5),
        "type_b_first_exam": lambda: max(random.gammavariate(3, 1), 0.5),
        "type_a_second_exam": lambda: max(random.gammavariate(4, 1), 0.5),
        "appointment_interval": lambda: max(random.uniform(13, 17), 1),
        "xray_probability": 0.80
    },
    {  # Doctor 3 (ID 2)
        "arrival": lambda: random.expovariate(1 / 12),
        "type_a_first_exam": lambda: max(random.gammavariate(2, 1), 0.5),
        "type_b_first_exam": lambda: max(random.gammavariate(4, 1), 0.5),
        "type_a_second_exam": lambda: max(random.gammavariate(6, 1), 0.5),
        "appointment_interval": lambda: max(random.uniform(11, 15), 1),
        "xray_probability": 0.78
    },
    {  # Doctor 4 (ID 3)
        "arrival": lambda: random.expovariate(1 / 11),
        "type_a_first_exam": lambda: max(random.gammavariate(2, 1), 0.5),
        "type_b_first_exam": lambda: max(random.gammavariate(3, 1), 0.5),
        "type_a_second_exam": lambda: max(random.gammavariate(4, 1), 0.5),
        "appointment_interval": lambda: max(random.uniform(8, 12), 1),
        "xray_probability": 0.60
    },
    {  # Doctor 5 (ID 4)
        "arrival": lambda: random.expovariate(1 / 13),
        "type_a_first_exam": lambda: max(random.gammavariate(2, 1), 0.5),
        "type_b_first_exam": lambda: max(random.gammavariate(3, 1), 0.5),
        "type_a_second_exam": lambda: max(random.gammavariate(5, 1), 0.5),
        "appointment_interval": lambda: max(random.uniform(9, 13), 1),
        "xray_probability": 0.65
    },
    {  # Doctor 6 (ID 5)
        "arrival": lambda: random.expovariate(1 / 10),
        "type_a_first_exam": lambda: max(random.gammavariate(2, 1), 0.5),
        "type_b_first_exam": lambda: max(random.gammavariate(5, 1), 0.5),
        "type_a_second_exam": lambda: max(random.gammavariate(2, 1), 0.5),
        "appointment_interval": lambda: max(random.uniform(11, 17), 1),
        "xray_probability": 0.90
    },
    {  # Doctor 7 (ID 6)
        "arrival": lambda: random.expovariate(1 / 14),
        "type_a_first_exam": lambda: max(random.gammavariate(2, 1), 0.5),
        "type_b_first_exam": lambda: max(random.gammavariate(5, 1), 0.5),
        "type_a_second_exam": lambda: max(random.gammavariate(3, 1), 0.5),
        "appointment_interval": lambda: max(random.uniform(10, 16), 1),
        "xray_probability": 0.87
    }
]

base_xray_service_time = lambda: max(random.gammavariate(1.25, 1.9), 1)

def get_actual_xray_service_time(env_now_time):
    service_time = base_xray_service_time()
    if env_now_time < LUNCH_END:
        return service_time * 1.25
    else:
        return service_time

# ----------- Statistics Tracking -----------
queue_lengths = defaultdict(list)
xray_room_queue_lengths = defaultdict(list)
timestamps = []
doctor_patient_count = [0] * NUM_DOCTORS
xray_patient_count = 0
doctor_second_exam_count = [0] * NUM_DOCTORS
appointment_arrival_count = 0
appointment_departure_count = 0
walk_in_arrival_count = 0
walk_in_departure_count = 0
total_patients_generated = 0
patients_currently_in_system = 0
appointment_actual_arrival_times = defaultdict(list)
appointment_scheduled_times = defaultdict(list)

# ----------- Doctor Lunch State -----------
doctor_is_on_lunch_break = [False] * NUM_DOCTORS

# ----------- Simulation Processes -----------

def manage_doctor_lunch_state(env, doctor_id):
    global doctor_is_on_lunch_break
    yield env.timeout(max(0, LUNCH_START - env.now))
    print(f"--- {env.now:.2f} - Dr {doctor_id+1} LUNCH BREAK PERIOD STARTED (will finish current patient) ---")
    doctor_is_on_lunch_break[doctor_id] = True
    yield env.timeout(max(0, LUNCH_END - env.now))
    print(f"--- {env.now:.2f} - Dr {doctor_id+1} LUNCH BREAK PERIOD ENDED (back to service) ---")
    doctor_is_on_lunch_break[doctor_id] = False

def patient(env, name, doctor_id, doctor_resource, is_appointment, needs_xray,
            actual_arrival_time, scheduled_arrival_time_for_priority=None):
    global doctor_patient_count, xray_patient_count, doctor_second_exam_count
    global appointment_departure_count, walk_in_departure_count
    global patients_currently_in_system, doctor_is_on_lunch_break
    global xray_resources

    if is_appointment:
        request_priority = scheduled_arrival_time_for_priority
        type_str_detail = f"Sched@{scheduled_arrival_time_for_priority:.2f}"
    else:
        request_priority = WALKIN_PRIORITY_OFFSET + actual_arrival_time
        type_str_detail = "Walk-in"

    type_str_base = f"{'Appointment' if is_appointment else 'Walk-in'} Type-{'A' if needs_xray else 'B'}"
    print(f"{actual_arrival_time:.2f} - {name} ({type_str_base}, {type_str_detail}) ARRIVED for Dr {doctor_id+1}.")

    try:
        # First Examination
        if doctor_is_on_lunch_break[doctor_id] and LUNCH_START <= env.now < LUNCH_END:
            if actual_arrival_time < LUNCH_END:
                wait_duration = LUNCH_END - env.now
                if wait_duration > 0:
                    print(f"{env.now:.2f} - {name} waiting until lunch ends ({LUNCH_END:.2f}) for Dr {doctor_id+1} (before 1st exam). Remaining: {wait_duration:.2f} min.")
                    yield env.timeout(wait_duration)
                    print(f"{env.now:.2f} - {name} continues after lunch for Dr {doctor_id+1} (1st exam).")

        print(f"{env.now:.2f} - {name} requests Dr {doctor_id+1} (Prio: {request_priority:.2f}). Queue: {len(doctor_resource.queue)}")
        start_wait_doc1 = env.now
        with doctor_resource.request(priority=request_priority) as req:
            yield req
            wait_time_doc1 = env.now - start_wait_doc1
            print(f"{env.now:.2f} - {name} 1st EXAM STARTED with Dr {doctor_id+1}. Wait: {wait_time_doc1:.2f} min. (Req Prio: {request_priority:.2f})")

            service_start_time_doc1 = env.now
            base_exam_time = doctor_configs[doctor_id]["type_a_first_exam"]() if needs_xray else doctor_configs[doctor_id]["type_b_first_exam"]()
            exam_time = base_exam_time
            speed_up_applied_doc1 = False
            if service_start_time_doc1 >= LUNCH_END:
                exam_time = base_exam_time * AFTERNOON_SPEEDUP_FACTOR
                speed_up_applied_doc1 = True

            yield env.timeout(exam_time)
            doctor_patient_count[doctor_id] += 1
            print(f"{env.now:.2f} - {name} 1st EXAM ENDED with Dr {doctor_id+1} (Duration: {exam_time:.2f} min {'[Sped up]' if speed_up_applied_doc1 else ''}).")

        # X-ray Process
        if needs_xray:
            xray_priority = request_priority
            print(f"{env.now:.2f} - {name} looking for an available X-ray room (Prio: {xray_priority:.2f}).")

            room_queue_lengths = [len(xr.queue) for xr in xray_resources]
            min_queue_len = min(room_queue_lengths)
            candidate_room_indices = [i for i, q_len in enumerate(room_queue_lengths) if q_len == min_queue_len]
            selected_xray_room_idx = random.choice(candidate_room_indices)
            chosen_xray_resource = xray_resources[selected_xray_room_idx]

            print(f"{env.now:.2f} - {name} requests X-ray Room {selected_xray_room_idx+1}. Room queue: {len(chosen_xray_resource.queue)}")
            start_wait_xray = env.now
            with chosen_xray_resource.request(priority=xray_priority) as req_xray:
                yield req_xray
                wait_time_xray = env.now - start_wait_xray
                print(f"{env.now:.2f} - {name} X-ray STARTED in Room {selected_xray_room_idx+1}. Wait: {wait_time_xray:.2f} min.")

                xray_time_val = get_actual_xray_service_time(env.now)
                yield env.timeout(xray_time_val)
                xray_patient_count += 1
                print(f"{env.now:.2f} - {name} X-ray ENDED in Room {selected_xray_room_idx+1} (Duration: {xray_time_val:.2f} min).")

            # Second Examination
            second_exam_priority = request_priority
            if doctor_is_on_lunch_break[doctor_id] and LUNCH_START <= env.now < LUNCH_END:
                if actual_arrival_time < LUNCH_END:
                    wait_duration_doc2 = LUNCH_END - env.now
                    if wait_duration_doc2 > 0:
                        print(f"{env.now:.2f} - {name} waiting until lunch ends ({LUNCH_END:.2f}) for Dr {doctor_id+1} (before 2nd exam). Remaining: {wait_duration_doc2:.2f} min.")
                        yield env.timeout(wait_duration_doc2)
                        print(f"{env.now:.2f} - {name} continues after lunch for Dr {doctor_id+1} (2nd exam).")

            print(f"{env.now:.2f} - {name} requests Dr {doctor_id+1} for 2nd exam (Prio: {second_exam_priority:.2f}). Queue: {len(doctor_resource.queue)}")
            start_wait_doc2 = env.now
            with doctor_resource.request(priority=second_exam_priority) as req_doc2:
                yield req_doc2
                wait_time_doc2 = env.now - start_wait_doc2
                print(f"{env.now:.2f} - {name} 2nd EXAM STARTED with Dr {doctor_id+1}. Wait: {wait_time_doc2:.2f} min.")

                service_start_time_doc2 = env.now
                base_second_exam_time = doctor_configs[doctor_id]["type_a_second_exam"]()
                second_exam_time = base_second_exam_time
                speed_up_applied_doc2 = False
                if service_start_time_doc2 >= LUNCH_END:
                    second_exam_time = base_second_exam_time * AFTERNOON_SPEEDUP_FACTOR
                    speed_up_applied_doc2 = True

                yield env.timeout(second_exam_time)
                doctor_second_exam_count[doctor_id] += 1
                print(f"{env.now:.2f} - {name} 2nd EXAM ENDED with Dr {doctor_id+1} (Duration: {second_exam_time:.2f} min {'[Sped up]' if speed_up_applied_doc2 else ''}).")

        # Departure
        departure_time = env.now
        if is_appointment:
            appointment_departure_count += 1
        else:
            walk_in_departure_count += 1
        print(f"{departure_time:.2f} - {name} DEPARTED. Time in system: {departure_time - actual_arrival_time:.2f} min.")
    finally:
        patients_currently_in_system -= 1
        if patients_currently_in_system < 0:
            print(f"!!!! ERROR !!!! Patient counter dropped below zero at {env.now:.2f} for {name}!")
            patients_currently_in_system = 0

def patient_generator(env, doctor_id, doctor_resource):
    global total_patients_generated, walk_in_arrival_count, patients_currently_in_system

    if APPOINTMENT_ONLY_DOCTOR_ID is not None and doctor_id == APPOINTMENT_ONLY_DOCTOR_ID:
        print(f"--- Dr {doctor_id+1} is APPOINTMENT-ONLY, so walk-in generator is not started for this doctor. ---")
        return

    patient_idx_walkin = 0
    while True:
        interarrival_time = doctor_configs[doctor_id]["arrival"]()
        potential_next_arrival = env.now + interarrival_time
        if potential_next_arrival >= WALKIN_CUTOFF_TIME:
            print(f"{env.now:.2f} - Dr {doctor_id+1} walk-in generator STOPPING. Next arrival ({potential_next_arrival:.2f}) would exceed cutoff ({WALKIN_CUTOFF_TIME}).")
            break
        yield env.timeout(max(0, interarrival_time))
        actual_arrival_time = env.now
        if actual_arrival_time >= WALKIN_CUTOFF_TIME:
            break

        patient_idx_walkin += 1
        total_patients_generated += 1
        walk_in_arrival_count += 1
        patients_currently_in_system += 1
        is_xray_needed = random.random() < doctor_configs[doctor_id]["xray_probability"]
        patient_name = f"Patient-WI-{doctor_id+1}-{patient_idx_walkin}"
        env.process(patient(env, patient_name, doctor_id, doctor_resource, False, is_xray_needed,
                            actual_arrival_time, scheduled_arrival_time_for_priority=None))

def appointment_generator(env, doctor_id, doctor_resource, scheduled_times_for_this_doctor):
    global total_patients_generated, appointment_arrival_count, patients_currently_in_system
    global appointment_actual_arrival_times

    if WALKIN_ONLY_DOCTOR_ID is not None and doctor_id == WALKIN_ONLY_DOCTOR_ID:
        print(f"--- Dr {doctor_id+1} is WALK-IN-ONLY, so appointment generator is not started for this doctor. ---")
        return

    patient_idx_appt = 0
    for scheduled_time in scheduled_times_for_this_doctor:
        patient_idx_appt += 1
        punctuality_deviation = random.uniform(UNIFORM_MIN_DEVIATION_MINUTES, UNIFORM_MAX_DEVIATION_MINUTES)
        actual_arrival_time_candidate = max(0, scheduled_time + punctuality_deviation)

        delay_until_actual_arrival = actual_arrival_time_candidate - env.now
        if delay_until_actual_arrival > 0:
            yield env.timeout(delay_until_actual_arrival)

        current_actual_arrival_time = env.now
        if current_actual_arrival_time >= SIM_TIME:
            continue

        appointment_actual_arrival_times[doctor_id].append(current_actual_arrival_time)
        total_patients_generated += 1
        appointment_arrival_count += 1
        patients_currently_in_system += 1
        is_xray_needed = random.random() < doctor_configs[doctor_id]["xray_probability"]
        patient_name = f"Patient-AP-{doctor_id+1}-{patient_idx_appt}"
        env.process(patient(env, patient_name, doctor_id, doctor_resource, True, is_xray_needed,
                            current_actual_arrival_time, scheduled_arrival_time_for_priority=scheduled_time))

def track_queues(env, doctors_list, stop_event_tracker):
    global timestamps, xray_resources, xray_room_queue_lengths
    while not stop_event_tracker.triggered:
        current_time_track = env.now
        timestamps.append(current_time_track)
        for i, d_res in enumerate(doctors_list):
            queue_lengths[i].append(len(d_res.queue))

        for room_idx in range(NUM_XRAY_ROOMS):
            xray_room_queue_lengths[room_idx].append(len(xray_resources[room_idx].queue))

        start_wait_track = env.now
        timeout_duration = QUEUE_TRACK_INTERVAL
        while timeout_duration > 0 and not stop_event_tracker.triggered:
            yield env.timeout(min(timeout_duration, 0.5))
            timeout_duration = QUEUE_TRACK_INTERVAL - (env.now - start_wait_track)

    current_time_track = env.now
    timestamps.append(current_time_track)
    for i, d_res in enumerate(doctors_list):
        queue_lengths[i].append(len(d_res.queue))
    for room_idx in range(NUM_XRAY_ROOMS):
        xray_room_queue_lengths[room_idx].append(len(xray_resources[room_idx].queue))
    print(f"{env.now:.2f} - Queue tracking stopped.")

def simulation_ender(env, stop_event_obj, doctors_list_for_check):
    global xray_resources
    if env.now < SIM_TIME:
        yield env.timeout(SIM_TIME - env.now)

    print(f"--- {env.now:.2f} - APPOINTMENT CUTOFF ({SIM_TIME} min) reached. Walk-ins stopped at {WALKIN_CUTOFF_TIME} min. Existing patients are finishing. ---")

    check_interval = 2
    while True:
        if patients_currently_in_system <= 0:
            all_doc_queues_empty = all(len(d.queue) == 0 for d in doctors_list_for_check)
            all_doc_resources_free = all(d.count == 0 for d in doctors_list_for_check)

            all_xray_queues_empty = all(len(xr.queue) == 0 for xr in xray_resources)
            all_xray_resources_free = all(xr.count == 0 for xr in xray_resources)

            if all_doc_queues_empty and all_doc_resources_free and all_xray_queues_empty and all_xray_resources_free:
                print(f"--- {env.now:.2f} - SYSTEM EMPTY (PatientCounter={patients_currently_in_system}, all queues/resources empty). Stopping simulation. ---")
                if not stop_event_obj.triggered:
                    stop_event_obj.succeed()
                break
            else:
                if int(env.now) % (check_interval * 5) < check_interval:
                    xray_q_details = [len(xr.queue) for xr in xray_resources]
                    xray_u_details = [xr.count for xr in xray_resources]
                    print(f"WARNING: {env.now:.2f} - PatientCounter={patients_currently_in_system} but queues/resources not empty. Re-checking. DocQ: {[len(d.queue) for d in doctors_list_for_check]}, XrayQ: {xray_q_details}, DocUsers: {[d.count for d in doctors_list_for_check]}, XrayUsers: {xray_u_details}")

        elif env.now > SIM_TIME + 3 * SIM_TIME:
            print(f"--- {env.now:.2f} - LONG RUN WARNING. PatientCounter={patients_currently_in_system}. Forcing stop. ---")
            if not stop_event_obj.triggered:
                stop_event_obj.succeed()
            break
        yield env.timeout(check_interval)

# ----------- Main Simulation Execution -----------
def main():
    global APPOINTMENT_ONLY_DOCTOR_ID
    global WALKIN_ONLY_DOCTOR_ID

    start_real_time = datetime.datetime.now()
    print(f"Starting simulation - Appointment cutoff: {SIM_TIME} min, Walk-in cutoff: {WALKIN_CUTOFF_TIME} min")
    print(f"Lunch break PERIOD: {LUNCH_START} - {LUNCH_END} min")
    print(f"AFTERNOON SPEEDUP (doctors): service times multiplied by {AFTERNOON_SPEEDUP_FACTOR:.0%} (time >= {LUNCH_END})")
    print(f"Doctors: {NUM_DOCTORS}, X-ray rooms: {NUM_XRAY_ROOMS}")
    print(f"X-ray service time: 1.25x slower before lunch end, normal after.")
    print(f"Appointment punctuality (Uniform): min dev={UNIFORM_MIN_DEVIATION_MINUTES} min, max dev={UNIFORM_MAX_DEVIATION_MINUTES} min")
    print(f"Random seed: {RANDOM_SEED}")

    if APPOINTMENT_ONLY_DOCTOR_ID is not None:
        if 0 <= APPOINTMENT_ONLY_DOCTOR_ID < NUM_DOCTORS:
            print(f"SPECIAL ROLE: Doctor {APPOINTMENT_ONLY_DOCTOR_ID + 1} is APPOINTMENT-ONLY.")
        else:
            print(f"WARNING: APPOINTMENT_ONLY_DOCTOR_ID ({APPOINTMENT_ONLY_DOCTOR_ID}) invalid. Disabling.")
            APPOINTMENT_ONLY_DOCTOR_ID = None

    if WALKIN_ONLY_DOCTOR_ID is not None:
        if 0 <= WALKIN_ONLY_DOCTOR_ID < NUM_DOCTORS:
            print(f"SPECIAL ROLE: Doctor {WALKIN_ONLY_DOCTOR_ID + 1} is WALK-IN-ONLY.")
        else:
            print(f"WARNING: WALKIN_ONLY_DOCTOR_ID ({WALKIN_ONLY_DOCTOR_ID}) invalid. Disabling.")
            WALKIN_ONLY_DOCTOR_ID = None

    if APPOINTMENT_ONLY_DOCTOR_ID is not None and APPOINTMENT_ONLY_DOCTOR_ID == WALKIN_ONLY_DOCTOR_ID:
        print(f"WARNING: APPOINTMENT_ONLY_DOCTOR_ID and WALKIN_ONLY_DOCTOR_ID assigned to same doctor ({APPOINTMENT_ONLY_DOCTOR_ID+1}). This doctor may see no patients. Please adjust IDs.")

    global doctors, xray_resources, doctor_is_on_lunch_break
    global queue_lengths, xray_room_queue_lengths, timestamps
    global doctor_patient_count, xray_patient_count, doctor_second_exam_count
    global appointment_arrival_count, appointment_departure_count
    global walk_in_arrival_count, walk_in_departure_count
    global total_patients_generated, patients_currently_in_system
    global appointment_actual_arrival_times, appointment_scheduled_times

    queue_lengths.clear(); xray_room_queue_lengths.clear(); timestamps.clear()
    appointment_actual_arrival_times.clear(); appointment_scheduled_times.clear()
    doctor_patient_count[:] = [0] * NUM_DOCTORS
    xray_patient_count = 0
    doctor_second_exam_count[:] = [0] * NUM_DOCTORS
    appointment_arrival_count = 0
    appointment_departure_count = 0
    walk_in_arrival_count = 0
    walk_in_departure_count = 0
    total_patients_generated = 0
    patients_currently_in_system = 0
    doctor_is_on_lunch_break[:] = [False] * NUM_DOCTORS
    xray_resources = []

    all_doctors_schedules = []
    print("\n--- Generating scheduled appointment times ---")
    for i in range(NUM_DOCTORS):
        if WALKIN_ONLY_DOCTOR_ID is not None and i == WALKIN_ONLY_DOCTOR_ID:
            all_doctors_schedules.append([])
            appointment_scheduled_times[i] = []
            print(f"  Dr {i+1} (Walk-in-only): no appointments scheduled.")
            continue

        doctor_schedule = []
        current_scheduled_time = 0
        while True:
            interval = doctor_configs[i]["appointment_interval"]()
            next_scheduled_time = current_scheduled_time + interval
            if next_scheduled_time < SIM_TIME:
                doctor_schedule.append(next_scheduled_time)
                current_scheduled_time = next_scheduled_time
            else:
                break

        all_doctors_schedules.append(doctor_schedule)
        appointment_scheduled_times[i] = doctor_schedule
        last_sched_time_str = f"{doctor_schedule[-1]:.2f}" if doctor_schedule else "None"
        print(f"  Dr {i+1}: {len(doctor_schedule)} appointments scheduled. Last scheduled: {last_sched_time_str} (Limit: {SIM_TIME})")
    print("--- Scheduling complete ---\n")

    env = simpy.Environment()
    stop_event = env.event()

    xray_resources.extend([simpy.PriorityResource(env, capacity=1) for _ in range(NUM_XRAY_ROOMS)])
    doctors = [simpy.PriorityResource(env, capacity=1) for _ in range(NUM_DOCTORS)]

    print("--- Starting patient generators ---")
    for i in range(NUM_DOCTORS):
        doctor_res = doctors[i]

        is_appointment_only = (APPOINTMENT_ONLY_DOCTOR_ID is not None and i == APPOINTMENT_ONLY_DOCTOR_ID)
        is_walkin_only = (WALKIN_ONLY_DOCTOR_ID is not None and i == WALKIN_ONLY_DOCTOR_ID)

        if is_appointment_only:
            if all_doctors_schedules[i]:
                env.process(appointment_generator(env, i, doctor_res, all_doctors_schedules[i]))
                print(f"  Dr {i+1}: APPOINTMENT-ONLY generator started.")
            else:
                print(f"  Dr {i+1} (Appointment-only): no scheduled appointments, generator not started.")
        elif is_walkin_only:
            env.process(patient_generator(env, i, doctor_res))
            print(f"  Dr {i+1}: WALK-IN-ONLY generator started.")
        else:
            env.process(patient_generator(env, i, doctor_res))
            if all_doctors_schedules[i]:
                env.process(appointment_generator(env, i, doctor_res, all_doctors_schedules[i]))
                print(f"  Dr {i+1}: BOTH appointment and walk-in generators started.")
            else:
                print(f"  Dr {i+1}: Walk-in generator started (no scheduled appointments).")

        env.process(manage_doctor_lunch_state(env, i))

    env.process(track_queues(env, doctors, stop_event))
    env.process(simulation_ender(env, stop_event, doctors))
    env.run(until=stop_event)

    end_real_time = datetime.datetime.now()
    print(f"\n--- Simulation finished ---")
    print(f"Total simulated time: {env.now:.2f} min")
    print(f"Wall-clock runtime: {end_real_time - start_real_time}")

    # Plotting
    min_len_ts = len(timestamps)
    if not timestamps or min_len_ts < 2:
        print("\nWARNING: Not enough timestamp data for plotting.")
    else:
        plot_timestamps = np.array(timestamps)

        plt.figure(figsize=(14, 7))
        for i in range(NUM_DOCTORS):
            q_data = queue_lengths.get(i, [])
            if len(q_data) < min_len_ts:
                last_val = q_data[-1] if q_data else 0
                q_data.extend([last_val] * (min_len_ts - len(q_data)))
            elif len(q_data) > min_len_ts:
                q_data = q_data[:min_len_ts]
            plt.plot(plot_timestamps, q_data, label=f'Dr {i+1} Queue', alpha=0.8)

        plt.axvline(x=LUNCH_START, color='grey', linestyle=':', linewidth=1, label=f'Lunch start ({LUNCH_START})')
        plt.axvline(x=LUNCH_END, color='dimgrey', linestyle='--', linewidth=1.2, label=f'Lunch end / Speedup ({LUNCH_END})')
        plt.axvline(x=WALKIN_CUTOFF_TIME, color='blue', linestyle='-.', linewidth=1.2, label=f'Walk-in cutoff ({WALKIN_CUTOFF_TIME})')
        plt.axvline(x=SIM_TIME, color='red', linestyle='-', linewidth=1.5, label=f'Appointment cutoff ({SIM_TIME})')

        title_suffix = ""
        if APPOINTMENT_ONLY_DOCTOR_ID is not None:
            title_suffix += f"\nDr {APPOINTMENT_ONLY_DOCTOR_ID+1} Appointment-only"
        if WALKIN_ONLY_DOCTOR_ID is not None:
            title_suffix += f", Dr {WALKIN_ONLY_DOCTOR_ID+1} Walk-in-only"

        plt.xlabel("Time (minutes)")
        plt.ylabel("Queue length")
        plt.title(
            f"Doctor Queue Lengths (Punctuality: Uniform [{UNIFORM_MIN_DEVIATION_MINUTES},{UNIFORM_MAX_DEVIATION_MINUTES}])\nSeed: {RANDOM_SEED}{title_suffix}"
        )
        plt.legend(fontsize='small', loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(14, 7))
        for room_idx in range(NUM_XRAY_ROOMS):
            xray_data_plot = xray_room_queue_lengths.get(room_idx, [])
            if len(xray_data_plot) < min_len_ts:
                last_val = xray_data_plot[-1] if xray_data_plot else 0
                xray_data_plot.extend([last_val] * (min_len_ts - len(xray_data_plot)))
            elif len(xray_data_plot) > min_len_ts:
                xray_data_plot = xray_data_plot[:min_len_ts]
            plt.plot(plot_timestamps, xray_data_plot, label=f"X-ray Room {room_idx+1} Queue", alpha=0.8)

        plt.axvline(x=LUNCH_START, color='grey', linestyle=':', linewidth=1, label=f'Lunch start ({LUNCH_START})')
        plt.axvline(x=LUNCH_END, color='dimgrey', linestyle='--', linewidth=1.2, label=f'Lunch end ({LUNCH_END})')
        plt.axvline(x=WALKIN_CUTOFF_TIME, color='blue', linestyle='-.', linewidth=1.2, label=f'Walk-in cutoff ({WALKIN_CUTOFF_TIME})')
        plt.axvline(x=SIM_TIME, color='red', linestyle='-', linewidth=1.5, label=f'Appointment cutoff ({SIM_TIME})')

        plt.xlabel("Time (minutes)")
        plt.ylabel("X-ray room queue length")
        plt.title(f"X-ray Room Queue Lengths ({NUM_XRAY_ROOMS} Rooms, 1.25x slower before lunch)\nSeed: {RANDOM_SEED}{title_suffix}")
        plt.legend(fontsize='small', loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.show()

    print("\n--- Patient Flow Statistics ---")
    print(f"Total scheduled appointments (all doctors): {sum(len(s) for s in all_doctors_schedules)}")
    print(f"Total generated patients (processes started): {total_patients_generated}")
    print(f"  Appointment arrivals (realized): {appointment_arrival_count}")
    print(f"  Walk-in arrivals: {walk_in_arrival_count}")

    total_departures = appointment_departure_count + walk_in_departure_count
    print(f"\nTotal completed patients: {total_departures}")
    print(f"  Appointment departures: {appointment_departure_count}")
    print(f"  Walk-in departures: {walk_in_departure_count}")

    calculated_remaining = total_patients_generated - total_departures
    print(f"Patients remaining at end (calculated): {calculated_remaining}")
    print(f"Patients remaining at end (counter): {patients_currently_in_system}")
    if calculated_remaining != patients_currently_in_system:
        print("!!! WARNING: Calculated remaining patients and counter value do not match. Please verify.")

    print("\n--- Doctor Activity Statistics ---")
    print("Total exams performed by doctors:")
    for i in range(NUM_DOCTORS):
        total_exams_by_doc = doctor_patient_count[i] + doctor_second_exam_count[i]
        print(f"  Doctor {i+1}: {total_exams_by_doc} (1st Exam: {doctor_patient_count[i]}, 2nd Exam: {doctor_second_exam_count[i]})")
    print(f"\nTotal X-ray patients (all rooms): {xray_patient_count} patients")

    # Doctor-level X-ray referral statistics (kept exactly as your logic, only translated)
    print("\n--- Doctor-level X-ray Referral Statistics ---")
    xray_column_header = "Sent to X-ray"
    print(f"{'Doctor':<10} | {'1st Exams':<12} | {xray_column_header:<14} | {'X-ray Rate (%)':<16} | {'Expected Rate (%)':<18}")
    print("-" * 80)
    for i in range(NUM_DOCTORS):
        total_first_exams_for_doc = doctor_patient_count[i]
        patients_to_xray_from_doc = doctor_second_exam_count[i]  # (your logic: second exam count as x-ray referrals)

        xray_referral_rate = 0
        if total_first_exams_for_doc > 0:
            xray_referral_rate = (patients_to_xray_from_doc / total_first_exams_for_doc) * 100

        expected_xray_prob_percent = doctor_configs[i]["xray_probability"] * 100

        print(f"Dr {i+1:<7} | {total_first_exams_for_doc:<12} | {patients_to_xray_from_doc:<14} | {xray_referral_rate:<16.2f} | {expected_xray_prob_percent:<18.2f}")

    total_first_exams_all_docs = sum(doctor_patient_count)
    total_second_exams_all_docs = sum(doctor_second_exam_count)

    if total_second_exams_all_docs != xray_patient_count:
        print(f"\nNote: Total second exams ({total_second_exams_all_docs}) and total X-ray patients ({xray_patient_count}) can differ.")
        print("This can happen if some patients do not return for the second exam before the simulation ends, or if processes are interrupted.")

    overall_xray_rate = 0
    if total_first_exams_all_docs > 0:
        overall_xray_rate = (total_second_exams_all_docs / total_first_exams_all_docs) * 100

    print("-" * 80)
    print(f"{'Total':<10} | {total_first_exams_all_docs:<12} | {total_second_exams_all_docs:<14} | {overall_xray_rate:<16.2f} | {'N/A':<18}")

if __name__ == "__main__":
    main()
