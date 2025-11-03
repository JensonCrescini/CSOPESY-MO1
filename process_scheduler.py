import threading, time, random, queue, sys
from collections import deque
from typing import List, Any, Dict

# Utility / Config

TICK_DURATION = 0.05
UINT16_MAX = 65535

DEFAULT_CONFIG = {
    "num-cpu": "2",
    "scheduler": "fcfs",
    "quantum-cycles": "3",
    "batch-process-freq": "5",
    "min-ins": "3",
    "max-ins": "8",
    "delays-per-exec": "0"
}

# Instruction Model

def clamp_uint16(v: int) -> int:
    if v < 0: return 0
    if v > UINT16_MAX: return UINT16_MAX
    return v


# Process

class Process:
    _id_counter = 1
    _id_lock = threading.Lock()

    def __init__(self, name: str, instr_count: int, delays_per_exec: int):
        with Process._id_lock:
            self.pid = Process._id_counter
            Process._id_counter += 1
        self.name = name
        self.instructions = self._generate_instructions(instr_count)
        self.pc_stack = [(self.instructions, 0, 1)]  # stack frames for FOR loops: (instr_list, index, repeat_remaining)
        self.vars: Dict[str,int] = {}
        self.finished = False
        self.logs: List[str] = []
        self.create_time = time.time()
        self.start_time = None
        self.end_time = None
        self.total_cpu_ticks = 0  # how many CPU ticks the process consumed
        self.delays_per_exec = delays_per_exec

    def _generate_instructions(self, n:int) -> List[Dict[str,Any]]:
        # Randomized instruction generation based on the PDF spec
        ins = []
        # always include at least one PRINT for demonstration
        for i in range(n):
            choice = random.choices(
                population=["PRINT","ADD","SUBTRACT","SLEEP","DECLARE","FOR"],
                weights=[0.25,0.2,0.15,0.15,0.15,0.1],
                k=1
            )[0]
            if choice == "PRINT":
                ins.append({"type":"PRINT", "args":[f"Hello world from {self.name}!"]})
            elif choice == "DECLARE":
                var_name = f"v{random.randint(0,9)}"
                value = random.randint(0, 100)
                ins.append({"type":"DECLARE", "args":[var_name, value]})
            elif choice == "ADD":
                var1 = f"v{random.randint(0,9)}"
                var2 = f"v{random.randint(0,9)}"
                # sometimes add immediate
                if random.random() < 0.4:
                    val = random.randint(0,50)
                    ins.append({"type":"ADD", "args":[var1, var2, val]})
                else:
                    var3 = f"v{random.randint(0,9)}"
                    ins.append({"type":"ADD", "args":[var1, var2, var3]})
            elif choice == "SUBTRACT":
                var1 = f"v{random.randint(0,9)}"
                var2 = f"v{random.randint(0,9)}"
                if random.random() < 0.4:
                    val = random.randint(0,50)
                    ins.append({"type":"SUBTRACT", "args":[var1, var2, val]})
                else:
                    var3 = f"v{random.randint(0,9)}"
                    ins.append({"type":"SUBTRACT", "args":[var1, var2, var3]})
            elif choice == "SLEEP":
                dur = random.randint(1,3)
                ins.append({"type":"SLEEP", "args":[dur]})
            elif choice == "FOR":
                # nested FOR up to depth 2 randomly; inner instructions small
                repeats = random.randint(2,4)
                body_len = random.randint(1,3)
                body = []
                for _ in range(body_len):
                    # small set inside for
                    r = random.choice(["PRINT","ADD","SLEEP"])
                    if r=="PRINT":
                        body.append({"type":"PRINT","args":[f"Loop msg from {self.name}"]})
                    elif r=="ADD":
                        v1 = f"v{random.randint(0,9)}"
                        v2 = f"v{random.randint(0,9)}"
                        body.append({"type":"ADD","args":[v1,v2,1]})
                    elif r=="SLEEP":
                        body.append({"type":"SLEEP","args":[1]})
                ins.append({"type":"FOR","body":body,"repeats":repeats})
        return ins

    def is_done(self):
        return self.finished

    def peek_next_instruction(self):
        # return the next instruction without advancing
        if not self.pc_stack:
            return None
        frame = self.pc_stack[-1]
        instr_list, idx, rep = frame
        if idx >= len(instr_list):
            return None
        return instr_list[idx]

    def step_instruction(self):
        """
        Execute exactly one *logical* instruction from the current frame.
        Returns ticks_consumed (int), output_lines (list of str)
        """
        if self.finished:
            return 0, []

        # Find current frame
        while self.pc_stack:
            instr_list, idx, rep = self.pc_stack[-1]
            if idx >= len(instr_list):
                # finish this frame, pop and decrement repeat
                self.pc_stack.pop()
                continue
            instr = instr_list[idx]
            # Advance the index in current frame now; if FOR and repeats remain we'll handle later
            self.pc_stack[-1] = (instr_list, idx+1, rep)
            # Execute instr
            instr_type = instr["type"]
            outputs = []
            ticks = max(1, self.delays_per_exec)  # each instruction consumes at least delays_per_exec ticks (busy)
            if instr_type == "PRINT":
                msg = instr["args"][0] if instr.get("args") else ""
                outputs.append(msg)
            elif instr_type == "DECLARE":
                var, val = instr["args"]
                self.vars[var] = clamp_uint16(int(val))
            elif instr_type == "ADD":
                var1 = instr["args"][0]
                op2 = instr["args"][1]
                op3 = instr["args"][2]
                v2 = self.vars.get(op2, 0) if isinstance(op2, str) else int(op2)
                v3 = self.vars.get(op3, 0) if isinstance(op3, str) else int(op3)
                res = clamp_uint16(v2 + v3)
                self.vars[var1] = res
            elif instr_type == "SUBTRACT":
                var1 = instr["args"][0]
                op2 = instr["args"][1]
                op3 = instr["args"][2]
                v2 = self.vars.get(op2, 0) if isinstance(op2, str) else int(op2)
                v3 = self.vars.get(op3, 0) if isinstance(op3, str) else int(op3)
                res = clamp_uint16(v2 - v3)
                self.vars[var1] = res
            elif instr_type == "SLEEP":
                dur = int(instr["args"][0]) if instr.get("args") else 1
                # SLEEP is in CPU ticks: it relinquishes CPU for dur ticks.
                # We'll treat SLEEP specially by returning negative ticks to indicate relinquish? Simpler:
                # Here, return ticks = dur * TICK_COST (simulate busy-wait? spec says relinquishes CPU)
                # We'll return 'sleep' as special: caller (CPU worker) will sleep for dur ticks and mark process not running.
                ticks = -int(dur)
            elif instr_type == "FOR":
                body = instr["body"]
                repeats = instr["repeats"]
                # Push a new frame for the FOR body repeated 'repeats' times
                # We emulate repeats by pushing (body, 0, repeats)
                self.pc_stack.append((body, 0, repeats))
                # executing FOR itself costs 1 tick
                ticks = max(1, self.delays_per_exec)
            else:
                outputs.append(f"[{self.name}] Unknown instruction {instr_type}")
            # After executing instruction, check if top frame finished and has repeats:
            # handle repeating frames (for loops)
            while self.pc_stack:
                top_list, top_idx, top_rep = self.pc_stack[-1]
                if top_idx >= len(top_list):
                    # finished one iteration of this frame
                    if top_rep > 1:
                        # decrement repeats and reset index
                        self.pc_stack[-1] = (top_list, 0, top_rep-1)
                        break
                    else:
                        # pop it
                        self.pc_stack.pop()
                        continue
                else:
                    break
            # If no frames left -> finished
            if not self.pc_stack:
                self.finished = True
                self.end_time = time.time()
            return ticks, outputs

        # if stack empty
        self.finished = True
        self.end_time = time.time()
        return 0, []

    def smi_summary(self):
        status = "Finished" if self.finished else "Running"
        return {
            "name": self.name,
            "pid": self.pid,
            "status": status,
            "instr_left": self._instructions_remaining_count(),
            "vars": dict(self.vars),
            "logs_tail": self.logs[-10:],
            "total_cpu_ticks": self.total_cpu_ticks
        }

    def _instructions_remaining_count(self):
        # approximate by summing frames
        total = 0
        for lst, idx, rep in self.pc_stack:
            total += max(0, len(lst) - idx) * rep
        return total


# Scheduler

class Scheduler:
    def __init__(self, config_path:str):
        self.load_config(config_path)
        # ready queue holds processes waiting to run
        # For FCFS: use deque; for RR: use deque but preempt and append left/right accordingly
        self.ready_queue = deque()
        self.running_procs = {}  # pid -> process currently assigned to a CPU
        self.finished_procs = {}  # pid -> process
        self.lock = threading.Lock()

        # CPU worker threads
        self.num_cpu = self.config_num_cpu
        self.cores = [None]*self.num_cpu  # store pid or None
        self.core_locks = [threading.Lock() for _ in range(self.num_cpu)]
        self.core_busy_ticks = [0]*self.num_cpu  # counts of ticks core was busy since start
        self.total_ticks = 0  # global tick counter since scheduler init
        self.ticker_running = False
        self.ticker_thread = None

        self.proc_gen_running = False
        self.proc_gen_thread = None

        # For stopping workers cleanly
        self.shutdown_flag = False

        # Worker threads pool:
        self.worker_threads = []
        for cid in range(self.num_cpu):
            t = threading.Thread(target=self.cpu_worker_loop, args=(cid,), daemon=True)
            self.worker_threads.append(t)
            t.start()

        # start ticker
        self.start_ticker()

    def load_config(self, path:str):
        cfg = DEFAULT_CONFIG.copy()
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"): continue
                    parts = line.split()
                    if len(parts) >= 2:
                        cfg[parts[0]] = parts[1]
        except FileNotFoundError:
            print(f"[initialize] Warning: {path} not found. Using defaults.")
        # parse
        self.config_num_cpu = int(cfg.get("num-cpu",DEFAULT_CONFIG["num-cpu"]))
        self.config_scheduler = cfg.get("scheduler","fcfs").lower()
        self.config_quantum = int(cfg.get("quantum-cycles", DEFAULT_CONFIG["quantum-cycles"]))
        self.config_batch_freq = int(cfg.get("batch-process-freq", DEFAULT_CONFIG["batch-process-freq"]))
        self.config_min_ins = int(cfg.get("min-ins", DEFAULT_CONFIG["min-ins"]))
        self.config_max_ins = int(cfg.get("max-ins", DEFAULT_CONFIG["max-ins"]))
        self.config_delays = int(cfg.get("delays-per-exec", DEFAULT_CONFIG["delays-per-exec"]))
        # clamp num-cpu to [1,128]
        if self.config_num_cpu < 1: self.config_num_cpu = 1
        if self.config_num_cpu > 128: self.config_num_cpu = 128

    # Ticker: increments global ticks

    def start_ticker(self):
        if self.ticker_running: return
        self.ticker_running = True
        self.ticker_thread = threading.Thread(target=self._ticker_loop, daemon=True)
        self.ticker_thread.start()

    def _ticker_loop(self):
        # increments total_ticks every TICK_DURATION
        while not self.shutdown_flag:
            time.sleep(TICK_DURATION)
            with self.lock:
                self.total_ticks += 1


    # Process generation (scheduler-start)

    def start_batch_generation(self):
        if self.proc_gen_running:
            print("scheduler already generating processes.")
            return
        self.proc_gen_running = True
        self.proc_gen_thread = threading.Thread(target=self._proc_gen_loop, daemon=True)
        self.proc_gen_thread.start()
        print("scheduler-start: batch generation started.")

    def stop_batch_generation(self):
        if not self.proc_gen_running:
            print("scheduler-stop: no batch generation running.")
            return
        self.proc_gen_running = False
        print("scheduler-stop: batch generation stopping...")
        if self.proc_gen_thread and self.proc_gen_thread.is_alive():
            self.proc_gen_thread.join(timeout=1.0)
        print("scheduler-stop: batch generation stopped.")


    def _proc_gen_loop(self):
        counter = 1
        last_tick = -1
        while not self.shutdown_flag:
            # Exit immediately if stop flag set
            if not self.proc_gen_running:
                break

            time.sleep(TICK_DURATION / 2)

            # Double-check flag after waking up
            if not self.proc_gen_running:
                break

            with self.lock:
                current_tick = self.total_ticks

            if last_tick == -1:
                last_tick = current_tick

            if (current_tick - last_tick) >= self.config_batch_freq:
                pname = f"p{counter:03d}"
                p = Process(pname, random.randint(self.config_min_ins, self.config_max_ins), self.config_delays)
                with self.lock:
                    self.ready_queue.append(p)
                counter += 1
                last_tick = current_tick


    # CPU worker loop

    def cpu_worker_loop(self, core_id:int):
        # Each worker repeatedly tries to fetch a process from ready_queue and run it according to scheduler
        while not self.shutdown_flag:
            proc = None
            with self.lock:
                # Fetch process according to scheduling algorithm
                if self.ready_queue:
                    # pick from left for FCFS, for RR also left
                    proc = self.ready_queue.popleft()
                    self.running_procs[proc.pid] = proc
                    self.cores[core_id] = proc.pid
                else:
                    self.cores[core_id] = None
            if not proc:
                time.sleep(TICK_DURATION/4)  # idle small wait
                continue

            # Mark start time if first run
            if proc.start_time is None:
                proc.start_time = time.time()

            # Run process instructions according to scheduling policy
            if self.config_scheduler == "fcfs":
                # Run until process finishes (or shutdown). Each instruction consumes delays-per-exec ticks.
                while not proc.finished and not self.shutdown_flag:
                    ticks, outputs = proc.step_instruction()
                    if ticks == 0:
                        # finished
                        break
                    if ticks < 0:
                        # SLEEP: relinquish CPU for -ticks CPU ticks (not busy)
                        sleep_ticks = -ticks
                        # release core: put back to ready queue after sleep
                        with self.lock:
                            self.cores[core_id] = None
                            # do not mark as finished; not in running_procs on core while sleeping
                            if proc.pid in self.running_procs:
                                del self.running_procs[proc.pid]
                        # sleep real time
                        time.sleep(sleep_ticks * TICK_DURATION)
                        # after sleeping re-enqueue (end of ready queue)
                        with self.lock:
                            if not proc.finished:
                                self.ready_queue.append(proc)
                        break
                    # busy-wait simulation: occupy core for 'ticks' ticks (counted as busy)
                    for _ in range(ticks):
                        if self.shutdown_flag: break
                        time.sleep(TICK_DURATION)
                        with self.lock:
                            self.core_busy_ticks[core_id] += 1
                            proc.total_cpu_ticks += 1
                    for line in outputs:
                        proc.logs.append(line)
                # if finished, move to finished list
                if proc.finished:
                    with self.lock:
                        self.cores[core_id] = None
                        if proc.pid in self.running_procs: del self.running_procs[proc.pid]
                        self.finished_procs[proc.pid] = proc
            else:
                # Round robin: run for at most quantum cycles (each instruction may take multiple ticks)
                remaining_quantum = self.config_quantum
                preempted = False
                while not proc.finished and remaining_quantum > 0 and not self.shutdown_flag:
                    ticks, outputs = proc.step_instruction()
                    if ticks == 0:
                        break
                    if ticks < 0:
                        # SLEEP: relinquish CPU for -ticks; similar to FCFS: sleep and requeue
                        sleep_ticks = -ticks
                        with self.lock:
                            self.cores[core_id] = None
                            if proc.pid in self.running_procs:
                                del self.running_procs[proc.pid]
                        time.sleep(sleep_ticks * TICK_DURATION)
                        with self.lock:
                            if not proc.finished:
                                self.ready_queue.append(proc)
                        preempted = True
                        break
                    # If instruction ticks exceed remaining quantum, we will consume only part and preempt
                    ticks_to_consume = min(ticks, remaining_quantum)
                    for _ in range(ticks_to_consume):
                        if self.shutdown_flag: break
                        time.sleep(TICK_DURATION)
                        with self.lock:
                            self.core_busy_ticks[core_id] += 1
                            proc.total_cpu_ticks += 1
                            remaining_quantum -= 1
                    # if instruction had leftover ticks (ticks - ticks_to_consume), we need to simulate that the instruction
                    # hasn't fully executed; but for simplicity we treat instruction as atomic regarding execution count:
                    # if ticks > ticks_to_consume, then we will continue it next time (by re-inserting a small "resume" instruction).
                    if ticks > ticks_to_consume:
                        # we simulate partial execution by reinserting an artificial instruction with reduced ticks.
                        # Easiest approach: reduce delays_per_exec temporarily — but simpler: put process back at front to resume.
                        # We'll requeue at front so it resumes soon.
                        with self.lock:
                            self.ready_queue.appendleft(proc)
                        preempted = True
                        break
                    for line in outputs:
                        proc.logs.append(line)
                # if process finished
                if proc.finished:
                    with self.lock:
                        self.cores[core_id] = None
                        if proc.pid in self.running_procs: del self.running_procs[proc.pid]
                        self.finished_procs[proc.pid] = proc
                elif not preempted and not proc.finished:
                    # quantum expired but process not finished -> preempt and append to ready queue end
                    with self.lock:
                        self.cores[core_id] = None
                        if proc.pid in self.running_procs: del self.running_procs[proc.pid]
                        self.ready_queue.append(proc)
                # else if preempted, it's already requeued

        # worker exiting
        return


    # CLI helpers

    def list_processes(self):
        with self.lock:
            running = [p for p in list(self.ready_queue)] + list(self.running_procs.values())
            finished = list(self.finished_procs.values())
            return running, finished

    def find_proc_by_name(self, name:str):
        with self.lock:
            # check ready queue
            for p in list(self.ready_queue):
                if p.name == name and not p.finished:
                    return p
            # running
            for p in list(self.running_procs.values()):
                if p.name == name and not p.finished:
                    return p
            # finished: per spec, finished should not be accessible
            return None

    def generate_util_report(self):
        with self.lock:
            total_time_ticks = max(1, self.total_ticks)
            total_core_ticks = sum(self.core_busy_ticks)
            max_possible_ticks = total_time_ticks * self.num_cpu
            utilization_pct = (total_core_ticks / max_possible_ticks)*100 if max_possible_ticks>0 else 0.0
            cores_used = sum(1 for c in self.cores if c is not None)
            cores_available = self.num_cpu - cores_used
            running = [p.name for p in list(self.ready_queue)] + [p.name for p in list(self.running_procs.values())]
            finished = [p.name for p in self.finished_procs.values()]
            per_proc_summary = []
            # combine running and finished data
            all_known = {p.pid:p for p in list(self.ready_queue) + list(self.running_procs.values()) + list(self.finished_procs.values())}
            for pid,p in all_known.items():
                per_proc_summary.append({
                    "name": p.name,
                    "pid": p.pid,
                    "status": "Finished" if p.finished else "Running",
                    "instr_left": p._instructions_remaining_count(),
                    "total_cpu_ticks": p.total_cpu_ticks
                })
            # write to csopesy-log.txt
            with open("csopesy-log.txt","w") as f:
                f.write("=== CSOPESY CPU UTILIZATION REPORT ===\n")
                f.write(f"Timestamp: {time.ctime()}\n")
                f.write(f"Total ticks elapsed: {total_time_ticks}\n")
                f.write(f"CPU cores configured: {self.num_cpu}\n")
                f.write(f"CPU utilization: {utilization_pct:.2f}% ({total_core_ticks} busy ticks of {max_possible_ticks} possible)\n")
                f.write(f"Cores used (now): {cores_used}\n")
                f.write(f"Cores available (now): {cores_available}\n")
                f.write("\nRunning processes:\n")
                for r in running:
                    f.write(f" - {r}\n")
                f.write("\nFinished processes:\n")
                for fn in finished:
                    f.write(f" - {fn}\n")
                f.write("\nPer-process summary:\n")
                for s in per_proc_summary:
                    f.write(f" - {s['name']} (pid {s['pid']}): {s['status']}, instr_left={s['instr_left']}, cpu_ticks={s['total_cpu_ticks']}\n")
            print("report-util: csopesy-log.txt generated.")
            return {
                "utilization_pct": utilization_pct,
                "cores_used": cores_used,
                "cores_available": cores_available,
                "running": running,
                "finished": finished,
                "per_process": per_proc_summary
            }

    def shutdown(self):
        self.shutdown_flag = True
        self.proc_gen_running = False
        self.ticker_running = False
        # allow workers to exit after checking flag
        print("Shutting down scheduler...")

# Screen Manager / CLI

class ScreenManager:
    def __init__(self, scheduler:Scheduler):
        self.scheduler = scheduler

    def handle_main(self, tokens:List[str]):
        # tokens[0] == "screen"
        if len(tokens) < 2:
            print("Usage: screen -s <name> | screen -ls | screen -r <name>")
            return
        flag = tokens[1]
        if flag == "-s":
            if len(tokens) != 3:
                print("Usage: screen -s <process_name>")
                return
            name = tokens[2]
            # create new process with provided name and put in ready queue
            # If name collides with existing active or finished -> deny
            if self._name_exists(name):
                print(f"Process {name} already exists (running or finished). Choose another name.")
                return
            # create process
            p = Process(name, random.randint(self.scheduler.config_min_ins, self.scheduler.config_max_ins), self.scheduler.config_delays)
            with self.scheduler.lock:
                self.scheduler.ready_queue.append(p)
            print(f"Created and attached to process {name}.")
            self._process_screen_loop(name)
        elif flag == "-ls":
            running, finished = self.scheduler.list_processes()
            print("=== screen -ls ===")
            print(f"Configured cores: {self.scheduler.num_cpu}")
            # cores used and available
            with self.scheduler.lock:
                cores_used = sum(1 for c in self.scheduler.cores if c is not None)
                cores_avail = self.scheduler.num_cpu - cores_used
            print(f"Cores used: {cores_used} | Cores available: {cores_avail}")
            print("\nRunning processes:")
            for p in running:
                print(f" - {p.name} (pid {p.pid}) {'[Finished]' if p.finished else ''}")
            print("\nFinished processes:")
            for p in finished:
                print(f" - {p.name} (pid {p.pid})")
        elif flag == "-r":
            if len(tokens) != 3:
                print("Usage: screen -r <process_name>")
                return
            name = tokens[2]
            proc = self.scheduler.find_proc_by_name(name)
            if not proc:
                print(f"Process {name} not found.")
                return
            print(f"Re-attaching to process {name}.")
            self._process_screen_loop(name)
        else:
            print("Invalid screen flags. Use -s, -ls, or -r.")

    def _name_exists(self, name:str):
        with self.scheduler.lock:
            for p in list(self.scheduler.ready_queue) + list(self.scheduler.running_procs.values()) + list(self.scheduler.finished_procs.values()):
                if p.name == name:
                    return True
            return False

    def _process_screen_loop(self, name:str):
        # interactive loop for attached screen; supports process-smi and exit
        while True:
            try:
                cmd = input(f"({name})> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nReturning to main menu.")
                break
            if not cmd:
                continue
            if cmd == "exit":
                print("Detaching and returning to main menu.")
                break
            elif cmd == "process-smi":
                proc = None
                # try to locate process in ready/running/finished lists
                with self.scheduler.lock:
                    for p in list(self.scheduler.ready_queue):
                        if p.name == name: proc = p; break
                    if not proc:
                        for p in list(self.scheduler.running_procs.values()):
                            if p.name == name: proc = p; break
                    if not proc:
                        # per spec: if finished, show "Finished!" after name, id, and logs
                        for p in list(self.scheduler.finished_procs.values()):
                            if p.name == name:
                                proc = p
                                break
                if not proc:
                    print(f"Process {name} not found.")
                    continue
                # print info
                info = proc.smi_summary()
                print(f"Process: {info['name']} (pid {info['pid']})")
                if proc.finished:
                    print("Finished!")
                print(f"Status: {info['status']}")
                print(f"Instructions remaining (approx): {info['instr_left']}")
                print(f"Total CPU ticks consumed: {info['total_cpu_ticks']}")
                print("Variables:")
                for k,v in info["vars"].items():
                    print(f"  {k} = {v}")
                print("Logs (last up to 10):")
                for ln in info["logs_tail"]:
                    print(f"  {ln}")
            else:
                print("Unknown screen command. Supported: process-smi, exit")


# Main CLI

def print_help():
    print("""
Available commands:
 - initialize           : load config.txt and start scheduler (required before other commands except exit)
 - exit                 : terminate console
 - screen -s <name>     : create new process and attach screen
 - screen -r <name>     : re-attach to an existing running process
 - screen -ls           : list running and finished processes, cores used/available
 - scheduler-start      : begin automatic dummy process generation
 - scheduler-stop       : stop automatic process generation
 - report-util          : generate csopesy-log.txt CPU utilization report
 - help                 : show this help
""")

def main():
    scheduler = None
    screen_manager = None
    initialized = False
    print("Welcome to CSOPESY OS Emulator (Python). Type 'help' for commands.")
    try:
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting console.")
                break
            if not line:
                continue
            tokens = line.split()
            cmd = tokens[0].lower()
            if cmd == "exit":
                print("Terminating console...")
                if scheduler:
                    scheduler.shutdown()
                    time.sleep(0.2)
                break
            if cmd == "help":
                print_help()
                continue
            if cmd == "initialize":
                # initialize scheduler with config.txt
                try:
                    scheduler = Scheduler("config.txt")
                    screen_manager = ScreenManager(scheduler)
                    initialized = True
                    print("initialize: configuration loaded and scheduler started.")
                    print(f" - num-cpu: {scheduler.config_num_cpu}")
                    print(f" - scheduler: {scheduler.config_scheduler}")
                    print(f" - quantum-cycles: {scheduler.config_quantum}")
                    print(f" - batch-process-freq: {scheduler.config_batch_freq} ticks")
                    print(f" - min-ins: {scheduler.config_min_ins} max-ins: {scheduler.config_max_ins}")
                    print(f" - delays-per-exec: {scheduler.config_delays} ticks")
                except Exception as e:
                    print("Error during initialize:", e)
                continue
            if not initialized:
                print("Error: Please run 'initialize' first (only 'initialize' and 'exit' available).")
                continue

            # recognized commands after initialized
            if cmd == "screen":
                screen_manager.handle_main(tokens)
            elif cmd == "scheduler-start":
                scheduler.start_batch_generation()
            elif cmd == "scheduler-stop":
                scheduler.stop_batch_generation()
            elif cmd == "report-util":
                scheduler.generate_util_report()
            else:
                print("Unknown command. Type 'help' to list commands.")
    finally:
        if scheduler:
            scheduler.shutdown()
        print("Console terminated.")

if __name__ == "__main__":
    main()
