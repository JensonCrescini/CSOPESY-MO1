#!/usr/bin/env python3
"""
CSOPESY MO2 - All-in-one implementation (single-file) for MO2 requirements.
Features:
 - FIFO page replacement demand pager
 - Backing store in `csopesy-backing-store.txt`
 - Processes have virtual memory ranges; first page is symbol table (64 bytes)
 - READ/WRITE memory instructions, DECLARE uses symbol table (max 32 vars)
 - process-smi and vmstat commands
 - screen -s <name> <memsize>
 - screen -c <name> <memsize> "<instr;...>"

Note: This single-file layout is for convenience. You can split into modules as needed.
"""

import threading, time, random, queue, sys, os
from collections import deque
from typing import List, Any, Dict, Optional, Tuple

# === Config / Defaults ===
TICK_DURATION = 0.05
UINT16_MAX = 65535
BACKING_STORE = "csopesy-backing-store.txt"

DEFAULT_CONFIG = {
    "num-cpu": "2",
    "scheduler": "fcfs",
    "quantum-cycles": "3",
    "batch-process-freq": "5",
    "min-ins": "3",
    "max-ins": "8",
    "delays-per-exec": "0",
    # MO2 specific defaults
    "max-overall-mem": "4096",
    "mem-per-frame": "256",
    "min-mem-per-proc": "256",
    "max-mem-per-proc": "1024"
}

# === Utilities ===

def clamp_uint16(v: int) -> int:
    if v < 0: return 0
    if v > UINT16_MAX: return UINT16_MAX
    return v

def is_power_of_two(n:int)->bool:
    return n>0 and (n & (n-1))==0

# === Memory Manager (FIFO pager) ===
class MemoryManager:
    def __init__(self, max_overall_mem:int, mem_per_frame:int):
        assert max_overall_mem % mem_per_frame == 0, "max_overall_mem must be multiple of mem_per_frame"
        self.max_mem = max_overall_mem
        self.frame_size = mem_per_frame
        self.num_frames = self.max_mem // self.frame_size
        # Frame table: frame_id -> (occupied:bool, pid, vpage)
        self.frames: List[Optional[Tuple[int,int]]] = [None]*self.num_frames
        # free frame list
        self.free_frames = deque(range(self.num_frames))
        # mapping: (pid, vpage) -> frame_id
        self.page_table: Dict[Tuple[int,int], int] = {}
        # FIFO queue of occupied frames for replacement
        self.fifo_queue = deque()
        # backing store: dictionary mapping (pid, vpage, offset) -> byte
        # We'll persist backing store in text file; but to keep it simple we'll maintain an in-memory dict and dump on writes.
        self.backing: Dict[Tuple[int,int,int], int] = {}
        # Stats
        self.paged_in = 0
        self.paged_out = 0
        # initialize backing file
        open(BACKING_STORE, 'a').close()

    def allocate_virtual_range(self, pid:int, size:int) -> int:
        """
        For simplicity, we use global virtual address space from 0..max_mem-1 and allocate by scanning for free contiguous virtual pages.
        Returns base virtual address (0..max_mem-1) or raises if cannot allocate.
        """
        pages_needed = (size + self.frame_size - 1)//self.frame_size
        # build occupancy of virtual pages by checking page_table keys
        occupied_vpages = set(vpage for (_pid,vpage) in self.page_table.keys() if _pid is not None)
        # naive: find first sequence of pages_needed free pages
        for start_page in range(0, self.max_mem // self.frame_size - pages_needed + 1):
            ok = True
            for p in range(start_page, start_page+pages_needed):
                # assume unallocated if no (pid,p) present in page_table for any pid
                # but page_table only contains mapped pages; to detect allocations we need another structure.
                pass
        # Instead of full allocator, we track virtual allocations separately.
        # Use self.allocations list of (pid, start_page, pages)
        if not hasattr(self, 'allocations'):
            self.allocations: List[Tuple[int,int,int]] = []
        max_vpages = self.max_mem // self.frame_size
        for start_page in range(0, max_vpages - pages_needed + 1):
            conflict = False
            for (_pid, sp, pn) in self.allocations:
                if not (start_page+pages_needed-1 < sp or start_page > sp+pn-1):
                    conflict = True
                    break
            if not conflict:
                self.allocations.append((pid, start_page, pages_needed))
                base_addr = start_page * self.frame_size
                return base_addr
        raise MemoryError("No contiguous virtual range available")

    def find_allocation(self, pid:int):
        if not hasattr(self,'allocations'): return None
        for (_pid, sp, pn) in self.allocations:
            if _pid==pid: return (sp,pn)
        return None

    def vaddr_to_vpage_offset(self, vaddr:int) -> Tuple[int,int]:
        if vaddr < 0 or vaddr >= self.max_mem:
            raise IndexError("virtual address out of range")
        vpage = vaddr // self.frame_size
        offset = vaddr % self.frame_size
        return vpage, offset

    def ensure_page_loaded(self, pid:int, vpage:int) -> int:
        """Ensure (pid,vpage) is loaded to a frame. Return frame id."""
        key = (pid, vpage)
        if key in self.page_table:
            return self.page_table[key]
        # Need to bring page from backing store
        # Find a free frame
        if self.free_frames:
            fid = self.free_frames.popleft()
        else:
            # FIFO replacement
            if not self.fifo_queue:
                raise MemoryError("No frames to evict")
            victim = self.fifo_queue.popleft()
            fid = victim
            # evict
            occ = self.frames[fid]
            if occ is not None:
                vpid, vvpage = occ
                # write back page to backing store (simulate by copying existing frame bytes to backing dict)
                # for simplicity, assume frame bytes are in self.frames_content
                if hasattr(self,'frames_content') and fid in self.frames_content:
                    page_bytes = self.frames_content[fid]
                    for off,b in enumerate(page_bytes):
                        if b is not None:
                            self.backing[(vpid,vvpage,off)] = b
                # remove mapping
                del self.page_table[(vpid,vvpage)]
                self.paged_out += 1
        # load page content from backing to frames_content
        if not hasattr(self,'frames_content'):
            self.frames_content = {}
        page_bytes = [None]*self.frame_size
        for off in range(self.frame_size):
            b = self.backing.get((pid,vpage,off), 0)
            page_bytes[off] = b
        self.frames_content[fid] = page_bytes
        self.frames[fid] = (pid, vpage)
        self.page_table[(pid,vpage)] = fid
        self.fifo_queue.append(fid)
        self.paged_in += 1
        return fid

    def read_uint16(self, pid:int, vaddr:int) -> int:
        vpage, offset = self.vaddr_to_vpage_offset(vaddr)
        # check allocation belongs to pid
        alloc = self.find_allocation(pid)
        if not alloc:
            raise MemoryError("Process has no allocation")
        sp, pn = alloc
        if vpage < sp or vpage >= sp+pn:
            raise PermissionError(f"Access violation: address 0x{vaddr:x} not in process {pid} range")
        fid = self.ensure_page_loaded(pid, vpage)
        # uint16 occupies 2 bytes; ensure bytes present possibly spanning frame boundary
        b0 = self.frames_content[fid][offset]
        if offset+1 < self.frame_size:
            b1 = self.frames_content[fid][offset+1]
        else:
            # need next page
            if vpage+1 >= sp+pn:
                b1 = 0
            else:
                fid2 = self.ensure_page_loaded(pid, vpage+1)
                b1 = self.frames_content[fid2][0]
        b0 = 0 if b0 is None else b0
        b1 = 0 if b1 is None else b1
        value = b0 | (b1<<8)
        return value

    def write_uint16(self, pid:int, vaddr:int, value:int):
        value = clamp_uint16(int(value))
        vpage, offset = self.vaddr_to_vpage_offset(vaddr)
        alloc = self.find_allocation(pid)
        if not alloc:
            raise MemoryError("Process has no allocation")
        sp, pn = alloc
        if vpage < sp or vpage >= sp+pn:
            raise PermissionError(f"Access violation: address 0x{vaddr:x} not in process {pid} range")
        fid = self.ensure_page_loaded(pid, vpage)
        # split to two bytes
        b0 = value & 0xff
        b1 = (value>>8) & 0xff
        self.frames_content[fid][offset] = b0
        if offset+1 < self.frame_size:
            self.frames_content[fid][offset+1] = b1
        else:
            # spill to next page
            if vpage+1 >= sp+pn:
                # writing beyond allocation -> access violation
                raise PermissionError(f"Access violation: address 0x{vaddr+1:x} not in process {pid} range")
            fid2 = self.ensure_page_loaded(pid, vpage+1)
            self.frames_content[fid2][0] = b1
        # also update backing store immediately
        off0 = offset
        self.backing[(pid,vpage,off0)] = b0
        if offset+1 < self.frame_size:
            self.backing[(pid,vpage,off0+1)] = b1
        else:
            self.backing[(pid,vpage+1,0)] = b1

    def dump_backing(self):
        # write readable backing to file for debugging
        with open(BACKING_STORE,'w') as f:
            for (pid,vpage,off),b in sorted(self.backing.items()):
                f.write(f"{pid}:{vpage}:{off}={b}\n")

    def stats(self):
        used_bytes = sum(1 for fid in range(self.num_frames) if self.frames[fid] is not None)*self.frame_size
        return {
            'total_memory': self.max_mem,
            'used_memory': used_bytes,
            'free_memory': self.max_mem - used_bytes,
            'frames': self.num_frames,
            'frame_size': self.frame_size,
            'pages_in': self.paged_in,
            'pages_out': self.paged_out
        }

# === Process Model ===
class Process:
    _id_counter = 1
    _id_lock = threading.Lock()

    def __init__(self, name:str, instr_list:List[Dict[str,Any]], mem_size:int, delays_per_exec:int, mm:MemoryManager):
        with Process._id_lock:
            self.pid = Process._id_counter
            Process._id_counter += 1
        self.name = name
        self.instructions = instr_list
        self.pc = 0
        self.vars: Dict[str,int] = {}
        self.var_count = 0
        self.finished = False
        self.logs: List[str] = []
        self.create_time = time.time()
        self.start_time = None
        self.end_time = None
        self.total_cpu_ticks = 0
        self.delays_per_exec = delays_per_exec
        self.mm = mm
        # allocate virtual range
        self.mem_size = mem_size
        self.base_addr = mm.allocate_virtual_range(self.pid, mem_size)
        self.pages = (mem_size + mm.frame_size -1)//mm.frame_size
        # Convention: symbol table placed at first frame of allocation; symbol table offset 0..63
        self.symbol_table_offset = 0
        self.symbol_table_size = 64
        self.max_vars = 32
        self.crashed = False
        self.crash_info = None

    def is_done(self):
        return self.finished or self.crashed or self.pc >= len(self.instructions)

    def smi_summary(self):
        status = "Finished" if self.finished else ("Crashed" if self.crashed else "Running")
        return {
            'name': self.name,
            'pid': self.pid,
            'status': status,
            'instr_left': max(0, len(self.instructions)-self.pc),
            'vars': dict(self.vars),
            'logs_tail': self.logs[-10:],
            'total_cpu_ticks': self.total_cpu_ticks,
            'mem_base': self.base_addr,
            'mem_size': self.mem_size
        }

    def step_instruction(self):
        if self.is_done():
            if not self.finished and not self.crashed:
                self.finished = True
                self.end_time = time.time()
            return 0, []
        instr = self.instructions[self.pc]
        self.pc += 1
        outputs = []
        ticks = max(1, self.delays_per_exec)
        itype = instr['type']
        try:
            if itype == 'PRINT':
                outputs.append(instr.get('args',[""])[0])
            elif itype == 'DECLARE':
                if self.var_count >= self.max_vars:
                    # ignore per spec
                    self.logs.append(f"Ignored DECLARE: var limit reached")
                else:
                    var, val = instr['args']
                    self.vars[var] = clamp_uint16(int(val))
                    self.var_count += 1
                    # store var to symbol table region in memory: symbol table is 64 bytes starting at base_addr
                    # each var uses 2 bytes; write at offset var_index*2
                    idx = self.var_count - 1
                    vaddr = self.base_addr + (self.symbol_table_offset + idx*2)
                    self.mm.write_uint16(self.pid, vaddr, self.vars[var])
            elif itype == 'ADD':
                dest, op2, op3 = instr['args']
                v2 = self.vars.get(op2, 0)
                v3 = self.vars.get(op3, 0) if isinstance(op3, str) else int(op3)
                res = clamp_uint16(v2 + v3)
                self.vars[dest] = res
                # if dest is symbol table variable, update backing
                if dest in self.vars:
                    # attempt to find index if it was declared earlier by name? Simple heuristic: no mapping; keep only in vars
                    pass
            elif itype == 'SUBTRACT':
                dest, op2, op3 = instr['args']
                v2 = self.vars.get(op2, 0)
                v3 = self.vars.get(op3, 0) if isinstance(op3, str) else int(op3)
                res = clamp_uint16(v2 - v3)
                self.vars[dest] = res
            elif itype == 'SLEEP':
                dur = int(instr['args'][0]) if instr.get('args') else 1
                ticks = -dur
            elif itype == 'READ':
                var, addrstr = instr['args']
                vaddr = int(addrstr,16)
                # read from memory
                val = self.mm.read_uint16(self.pid, vaddr)
                self.vars[var] = val
                outputs.append(f"READ {var} = {val}")
            elif itype == 'WRITE':
                addrstr, val_or_var = instr['args']
                vaddr = int(addrstr,16)
                if isinstance(val_or_var, str) and val_or_var in self.vars:
                    val = self.vars[val_or_var]
                else:
                    val = int(val_or_var)
                self.mm.write_uint16(self.pid, vaddr, val)
                outputs.append(f"WROTE 0x{vaddr:x} <- {val}")
            elif itype == 'FOR':
                repeats = instr['repeats']
                body = instr['body']
                # expand body inline by inserting repeats copies (simple approach)
                insert = []
                for _ in range(repeats):
                    for b in body:
                        insert.append(b.copy())
                # insert at current pc
                self.instructions[self.pc:self.pc] = insert
            else:
                outputs.append(f"Unknown instr {itype}")
        except PermissionError as pe:
            # crash
            self.crashed = True
            ts = time.strftime('%H:%M:%S', time.localtime())
            self.crash_info = (ts, str(pe))
            outputs.append(f"Process {self.name} crashed: {pe}")
            return 0, outputs
        except Exception as e:
            self.crashed = True
            ts = time.strftime('%H:%M:%S', time.localtime())
            self.crash_info = (ts, str(e))
            outputs.append(f"Process {self.name} crashed: {e}")
            return 0, outputs
        # mark finished if pc beyond
        if self.pc >= len(self.instructions):
            self.finished = True
            self.end_time = time.time()
        return ticks, outputs

# === Scheduler (updated) ===
class Scheduler:
    def __init__(self, config_path:str):
        self.load_config(config_path)
        # memory manager
        self.mm = MemoryManager(self.config_max_overall_mem, self.config_mem_per_frame)
        # ready queue holds processes waiting to run
        self.ready_queue = deque()
        self.running_procs = {}
        self.finished_procs = {}
        self.lock = threading.Lock()

        self.num_cpu = self.config_num_cpu
        self.cores = [None]*self.num_cpu
        self.core_busy_ticks = [0]*self.num_cpu
        self.total_ticks = 0
        self.ticker_running = False
        self.ticker_thread = None

        self.proc_gen_running = False
        self.proc_gen_thread = None
        
        # === VMSTAT FIELDS (static simulated values) ===
        self.total_memory = self.mm.max_mem                  # TOTAL RAM = max_overall_mem
        self.used_memory = 0                                 # updated dynamically below
        self.free_memory = self.total_memory                 # total - used

        # Active / inactive memory (simulated)
        self.active_memory = int(self.total_memory * 0.40)
        self.inactive_memory = int(self.total_memory * 0.20)
        self.buffer_memory = int(self.total_memory * 0.02)
        self.swap_cache = int(self.total_memory * 0.05)

        # Swap (simulated)
        self.total_swap = self.total_memory * 2
        self.free_swap = self.total_swap
        self.used_swap = 0

        # CPU ticks (simulated)
        self.user_cpu_ticks = 0
        self.nice_user_cpu_ticks = 0
        self.system_cpu_ticks = 0
        self.idle_cpu_ticks = 0
        self.iowait_cpu_ticks = 0
        self.irq_cpu_ticks = 0
        self.softirq_cpu_ticks = 0
        self.steal_cpu_ticks = 0

        # Paging from memory manager
        self.pages_paged_in = 0
        self.pages_paged_out = 0

        # Misc counters
        self.context_switches = 0
        self.boot_time = int(time.time()) - random.randint(10_000, 50_000)
        self.forks = 0


        self.shutdown_flag = False
        self.worker_threads = []
        for cid in range(self.num_cpu):
            t = threading.Thread(target=self.cpu_worker_loop, args=(cid,), daemon=True)
            self.worker_threads.append(t)
            t.start()
        self.start_ticker()

    def load_config(self, path:str):
        cfg = DEFAULT_CONFIG.copy()
        try:
            with open(path,'r') as f:
                for line in f:
                    line=line.strip()
                    if not line or line.startswith('#'): continue
                    parts = line.split()
                    if len(parts)>=2:
                        cfg[parts[0]] = parts[1]
        except FileNotFoundError:
            print(f"[initialize] {path} not found, using defaults")
        self.config_num_cpu = int(cfg.get('num-cpu'))
        self.config_scheduler = cfg.get('scheduler').lower()
        self.config_quantum = int(cfg.get('quantum-cycles'))
        self.config_batch_freq = int(cfg.get('batch-process-freq'))
        self.config_min_ins = int(cfg.get('min-ins'))
        self.config_max_ins = int(cfg.get('max-ins'))
        self.config_delays = int(cfg.get('delays-per-exec'))
        # MO2 params
        self.config_max_overall_mem = int(cfg.get('max-overall-mem'))
        self.config_mem_per_frame = int(cfg.get('mem-per-frame'))
        self.config_min_mem_per_proc = int(cfg.get('min-mem-per-proc'))
        self.config_max_mem_per_proc = int(cfg.get('max-mem-per-proc'))
        # clamp
        if self.config_num_cpu < 1: self.config_num_cpu = 1

    def start_ticker(self):
        if self.ticker_running: return
        self.ticker_running = True
        self.ticker_thread = threading.Thread(target=self._ticker_loop, daemon=True)
        self.ticker_thread.start()

    def _ticker_loop(self):
        while not self.shutdown_flag:
            time.sleep(TICK_DURATION)
            with self.lock:
                self.total_ticks += 1

    # batch generation
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
            if not self.proc_gen_running:
                break
            time.sleep(TICK_DURATION/2)
            if not self.proc_gen_running:
                break
            with self.lock:
                current_tick = self.total_ticks
            if last_tick == -1: last_tick = current_tick
            if (current_tick - last_tick) >= self.config_batch_freq:
                pname = f"p{counter:03d}"
                # random mem size between min and max but must be power of two per MO2
                m = random.choice([64,128,256,512,1024,2048])
                m = max(self.config_min_mem_per_proc, min(m, self.config_max_mem_per_proc))
                instr_count = random.randint(self.config_min_ins, self.config_max_ins)
                instrs = self._gen_instrs(instr_count, pname)
                p = Process(pname, instrs, m, self.config_delays, self.mm)
                with self.lock:
                    self.ready_queue.append(p)
                counter += 1
                last_tick = current_tick

    def _gen_instrs(self, n:int, pname:str):
        ins = []
        for i in range(n):
            choice = random.choices(["PRINT","DECLARE","ADD","SUBTRACT","SLEEP","READ","WRITE","FOR"], weights=[0.2,0.15,0.15,0.1,0.1,0.1,0.1,0.1])[0]
            if choice=="PRINT":
                ins.append({'type':'PRINT','args':[f"Hello from {pname}"]})
            elif choice=='DECLARE':
                var = f"v{random.randint(0,9)}"
                val = random.randint(0,100)
                ins.append({'type':'DECLARE','args':[var,val]})
            elif choice=='ADD':
                v1=f"v{random.randint(0,9)}"; v2=f"v{random.randint(0,9)}"; v3=f"v{random.randint(0,9)}"
                ins.append({'type':'ADD','args':[v1,v2,v3]})
            elif choice=='SUBTRACT':
                v1=f"v{random.randint(0,9)}"; v2=f"v{random.randint(0,9)}"; v3=f"v{random.randint(0,9)}"
                ins.append({'type':'SUBTRACT','args':[v1,v2,v3]})
            elif choice=='SLEEP':
                ins.append({'type':'SLEEP','args':[random.randint(1,3)]})
            elif choice=='READ':
                # pick random address in process allocation later; use placeholder 0x0 which will likely cause fault and load
                addr = f"0x{random.randint(0, self.mm.max_mem-1):x}"
                ins.append({'type':'READ','args':[f"r{random.randint(0,9)}", addr]})
            elif choice=='WRITE':
                addr = f"0x{random.randint(0,self.mm.max_mem-1):x}"
                val = random.randint(0,100)
                ins.append({'type':'WRITE','args':[addr, val]})
            elif choice=='FOR':
                repeats = random.randint(2,3)
                body = [{'type':'PRINT','args':[f"Loop {pname}"]}]
                ins.append({'type':'FOR','repeats':repeats,'body':body})
        return ins

    def cpu_worker_loop(self, core_id:int):
        while not self.shutdown_flag:
            proc = None
            with self.lock:
                if self.ready_queue:
                    proc = self.ready_queue.popleft()
                    self.running_procs[proc.pid] = proc
                    self.cores[core_id] = proc.pid
                else:
                    self.cores[core_id] = None
            if not proc:
                time.sleep(TICK_DURATION/4)
                continue
            if proc.start_time is None:
                proc.start_time = time.time()
            if self.config_scheduler == 'fcfs':
                while not proc.is_done() and not self.shutdown_flag:
                    ticks, outputs = proc.step_instruction()
                    if ticks == 0:
                        break
                    if ticks < 0:
                        sleep_ticks = -ticks
                        with self.lock:
                            self.cores[core_id] = None
                            if proc.pid in self.running_procs: del self.running_procs[proc.pid]
                        time.sleep(sleep_ticks * TICK_DURATION)
                        with self.lock:
                            if not proc.is_done(): self.ready_queue.append(proc)
                        break
                    for _ in range(ticks):
                        if self.shutdown_flag: break
                        time.sleep(TICK_DURATION)
                        with self.lock:
                            self.core_busy_ticks[core_id] += 1
                            proc.total_cpu_ticks += 1
                    for ln in outputs:
                        proc.logs.append(ln)
                if proc.is_done():
                    with self.lock:
                        self.cores[core_id] = None
                        if proc.pid in self.running_procs: del self.running_procs[proc.pid]
                        self.finished_procs[proc.pid] = proc
            else:
                # rr
                remaining = self.config_quantum
                preempted = False
                while not proc.is_done() and remaining>0 and not self.shutdown_flag:
                    ticks, outputs = proc.step_instruction()
                    if ticks==0: break
                    if ticks<0:
                        st = -ticks
                        with self.lock:
                            self.cores[core_id] = None
                            if proc.pid in self.running_procs: del self.running_procs[proc.pid]
                        time.sleep(st * TICK_DURATION)
                        with self.lock:
                            if not proc.is_done(): self.ready_queue.append(proc)
                        preempted = True
                        break
                    to_consume = min(ticks, remaining)
                    for _ in range(to_consume):
                        if self.shutdown_flag: break
                        time.sleep(TICK_DURATION)
                        with self.lock:
                            self.core_busy_ticks[core_id] += 1
                            proc.total_cpu_ticks += 1
                            remaining -= 1
                    if ticks > to_consume:
                        with self.lock:
                            self.ready_queue.appendleft(proc)
                        preempted = True
                        break
                    for ln in outputs:
                        proc.logs.append(ln)
                if proc.is_done():
                    with self.lock:
                        self.cores[core_id] = None
                        if proc.pid in self.running_procs: del self.running_procs[proc.pid]
                        self.finished_procs[proc.pid] = proc
                elif not preempted:
                    with self.lock:
                        self.cores[core_id] = None
                        if proc.pid in self.running_procs: del self.running_procs[proc.pid]
                        self.ready_queue.append(proc)
        return

    # CLI helpers
    def list_processes(self):
        with self.lock:
            running = [p for p in list(self.ready_queue)] + list(self.running_procs.values())
            finished = list(self.finished_procs.values())
            return running, finished

    def find_proc_by_name(self, name:str):
        with self.lock:
            for p in list(self.ready_queue):
                if p.name==name and not p.is_done(): return p
            for p in list(self.running_procs.values()):
                if p.name==name and not p.is_done(): return p
            for p in list(self.finished_procs.values()):
                if p.name==name: return p
            return None

    def generate_util_report(self):
        with self.lock:
            total_time_ticks = max(1, self.total_ticks)
            total_core_ticks = sum(self.core_busy_ticks)
            max_possible = total_time_ticks * self.num_cpu
            utilization = (total_core_ticks/max_possible)*100 if max_possible>0 else 0.0
            running = [p.name for p in list(self.ready_queue)] + [p.name for p in list(self.running_procs.values())]
            finished = [p.name for p in self.finished_procs.values()]
            per_proc = []
            all_known = {p.pid:p for p in list(self.ready_queue)+list(self.running_procs.values())+list(self.finished_procs.values())}
            for pid,p in all_known.items():
                per_proc.append({'name':p.name,'pid':p.pid,'status':('Finished' if p.finished else ('Crashed' if p.crashed else 'Running')),'instr_left':max(0,len(p.instructions)-p.pc),'total_cpu_ticks':p.total_cpu_ticks})
            # write report
            with open('csopesy-log.txt','w') as f:
                f.write('=== CSOPESY CPU UTILIZATION REPORT ===\n')
                f.write(f"Timestamp: {time.ctime()}\n")
                f.write(f"Total ticks elapsed: {total_time_ticks}\n")
                f.write(f"CPU cores configured: {self.num_cpu}\n")
                f.write(f"CPU utilization: {utilization:.2f}% ({total_core_ticks} busy ticks of {max_possible} possible)\n")
                f.write('\nRunning processes:\n')
                for r in running: f.write(f" - {r}\n")
                f.write('\nFinished processes:\n')
                for fn in finished: f.write(f" - {fn}\n")
                f.write('\nPer-process summary:\n')
                for s in per_proc:
                    f.write(f" - {s['name']} (pid {s['pid']}): {s['status']}, instr_left={s['instr_left']}, cpu_ticks={s['total_cpu_ticks']}\n")
            print('report-util: csopesy-log.txt generated.')
            return {'utilization_pct':utilization,'running':running,'finished':finished,'per_process':per_proc}

    def shutdown(self):
        self.shutdown_flag = True
        self.proc_gen_running = False
        self.ticker_running = False
        print('Shutting down scheduler...')

# === Screen Manager / CLI ===
class ScreenManager:
    def __init__(self, scheduler:Scheduler):
        self.scheduler = scheduler

    def handle_main(self, tokens:List[str]):
        if len(tokens)<2:
            print("Usage: screen -s <name> <mem> | screen -c <name> <mem> \"<instrs>\" | screen -ls | screen -r <name>")
            return
        flag = tokens[1]
        if flag == '-s':
            if len(tokens)!=4:
                print('Usage: screen -s <process_name> <process_memory_size>')
                return
            name = tokens[2]; mem = int(tokens[3])
            if not is_power_of_two(mem) or mem < 64 or mem > self.scheduler.config_max_overall_mem:
                print('invalid memory allocation')
                return
            if self._name_exists(name):
                print(f"Process {name} already exists.")
                return
            instrs = self.scheduler._gen_instrs(random.randint(self.scheduler.config_min_ins,self.scheduler.config_max_ins), name)
            p = Process(name, instrs, mem, self.scheduler.config_delays, self.scheduler.mm)
            with self.scheduler.lock:
                self.scheduler.ready_queue.append(p)
            print(f'Created and attached to process {name}.')
            self._process_screen_loop(name)
        elif flag == '-c':
            if len(tokens)<5:
                print('Usage: screen -c <name> <mem> "<instrs>"')
                return
            name = tokens[2]; mem = int(tokens[3]); instr_str = ' '.join(tokens[4:]).strip('"')
            if len(instr_str)==0 or len(instr_str.split(';'))>50:
                print('invalid command')
                return
            if not is_power_of_two(mem) or mem<64 or mem>self.scheduler.config_max_overall_mem:
                print('invalid memory allocation')
                return
            if self._name_exists(name):
                print(f"Process {name} already exists.")
                return
            instrs = parse_instruction_string(instr_str)
            p = Process(name, instrs, mem, self.scheduler.config_delays, self.scheduler.mm)
            with self.scheduler.lock:
                self.scheduler.ready_queue.append(p)
            print(f'Created and attached to process {name} (user instructions).')
            self._process_screen_loop(name)
        elif flag == '-ls':
            running, finished = self.scheduler.list_processes()
            print('=== screen -ls ===')
            print(f'Configured cores: {self.scheduler.num_cpu}')
            with self.scheduler.lock:
                cores_used = sum(1 for c in self.scheduler.cores if c is not None)
                cores_avail = self.scheduler.num_cpu - cores_used
            print(f'Cores used: {cores_used} | Cores available: {cores_avail}')
            print('\nRunning processes:')
            for p in running:
                print(f" - {p.name} (pid {p.pid}) {'[Finished]' if p.finished else ''}")
            print('\nFinished processes:')
            for p in finished:
                print(f" - {p.name} (pid {p.pid})")
        elif flag == '-r':
            if len(tokens)!=3:
                print('Usage: screen -r <process_name>')
                return
            name = tokens[2]
            proc = self.scheduler.find_proc_by_name(name)
            if not proc:
                print(f'Process {name} not found.')
                return
            if proc.crashed:
                ts, reason = proc.crash_info
                print(f"Process {name} shut down due to memory access violation error that occurred at {ts}. {reason}")
                return
            print(f'Re-attaching to process {name}.')
            self._process_screen_loop(name)
        else:
            print('Invalid screen flags.')

    def _name_exists(self, name:str):
        with self.scheduler.lock:
            for p in list(self.scheduler.ready_queue)+list(self.scheduler.running_procs.values())+list(self.scheduler.finished_procs.values()):
                if p.name==name: return True
            return False

    def _process_screen_loop(self, name:str):
        while True:
            try:
                cmd = input(f"({name})> ").strip()
            except (EOFError, KeyboardInterrupt):
                print('\nReturning to main menu.')
                break
            if not cmd: continue
            if cmd=='exit':
                print('Detaching and returning to main menu.'); break
            if cmd=='process-smi':
                proc = self.scheduler.find_proc_by_name(name)
                if not proc:
                    print(f'Process {name} not found.'); continue
                info = proc.smi_summary()
                print(f"Process: {info['name']} (pid {info['pid']})")
                if proc.finished: print('Finished!')
                if proc.crashed: print('Crashed!')
                print(f"Status: {info['status']}")
                print(f"Instructions remaining (approx): {info['instr_left']}")
                print(f"Total CPU ticks consumed: {info['total_cpu_ticks']}")
                print('Variables:')
                for k,v in info['vars'].items(): print(f"  {k} = {v}")
                print('Logs (last up to 10):')
                for ln in info['logs_tail']: print(f"  {ln}")
            else:
                print('Unknown screen command. Supported: process-smi, exit')

# === Helper: instruction parser for screen -c ===
def parse_instruction_string(s:str)->List[Dict[str,Any]]:
    parts = [p.strip() for p in s.split(';') if p.strip()]
    res = []
    for p in parts:
        # naive parser: split by spaces, handle PRINT with parentheses
        if p.upper().startswith('PRINT'):
            # extract inside parentheses or after space
            start = p.find('(')
            if start!=-1:
                inside = p[start+1:p.rfind(')')]
                res.append({'type':'PRINT','args':[inside.strip('"')]})
            else:
                toks = p.split(None,1)
                arg = toks[1] if len(toks)>1 else ''
                res.append({'type':'PRINT','args':[arg.strip('"')]})
        else:
            toks = p.split()
            cmd = toks[0].upper()
            args = toks[1:]
            res.append({'type':cmd,'args':args})
    return res

# === vmstat / process-smi commands ===
def vmstat(sched: Scheduler):
    mm = sched.mm
    mem = mm.stats()   # real memory stats

    print("vmstat -s")

    # MEMORY
    print(f"{mem['total_memory'] // 1024:12} K total memory")
    print(f"{mem['used_memory'] // 1024:12} K used memory")
    
    # Active/inactive memory (approximation)
    active = int(mem['used_memory'] * 0.60)
    inactive = int(mem['used_memory'] * 0.40)
    print(f"{active // 1024:12} K active memory")
    print(f"{inactive // 1024:12} K inactive memory")

    # Free memory
    print(f"{mem['free_memory'] // 1024:12} K free memory")

    # Buffers + swap cache (simulated but proportional to memory)
    print(f"{(mem['used_memory'] // 20) // 1024:12} K buffer memory")
    print(f"{(mem['used_memory'] // 30) // 1024:12} K swap cache")

    # SWAP (You can define real swap later)
    total_swap = mem['total_memory'] * 2
    free_swap  = total_swap  # no swapping implemented yet

    print(f"{total_swap // 1024:12} K total swap")
    print(f"{free_swap // 1024:12} K free swap")

    # CPU TICKS (real MO2 values)
    print(f"{sched.core_busy_ticks[0] + sched.core_busy_ticks[1]:12}    user cpu ticks")
    print(f"{0:12}    nice user cpu ticks")
    print(f"{0:12}    system cpu ticks")
    
    idle = max(0, sched.total_ticks - sum(sched.core_busy_ticks))
    print(f"{idle:12}    idle cpu ticks")

    print(f"{0:12}    IO-wait cpu ticks")
    print(f"{0:12}    IRQ cpu ticks")
    print(f"{0:12}    softirq cpu ticks")
    print(f"{0:12}    stolen cpu ticks")

    # Paging from FIFO MemoryManager
    print(f"{mm.paged_in:12}    pages paged in")
    print(f"{mm.paged_out:12}    pages paged out")

    # Scheduler context switches (approx)
    ctx = sum(sched.core_busy_ticks)
    print(f"{ctx:12}    CPU context switches")

    # Boot time (real timestamp)
    print(f"{sched.boot_time:12}    boot time")

    # Forks = number of processes created
    forks = Process._id_counter - 1
    print(f"{forks:12}    forks")

        
def print_help():
    print('''
Available commands:
 - initialize           : load config.txt and start scheduler (required before other commands except exit)
 - exit                 : terminate console
 - screen -s <name> <mem>     : create new process and attach screen
 - screen -c <name> <mem> "<instructions>"     : create process with user instructions
 - screen -r <name>     : re-attach to an existing running process
 - screen -ls           : list running and finished processes, cores used/available
 - scheduler-start      : begin automatic dummy process generation
 - scheduler-stop       : stop automatic process generation
 - report-util          : generate csopesy-log.txt CPU utilization report
 - process-smi <name>   : show process memory and summary
 - vmstat               : show memory and paging stats
 - help                 : show this help
''')


def main():
    scheduler = None
    screen_manager = None
    initialized = False
    print('Welcome to CSOPESY MO2 Emulator. Type help for commands.')
    try:
        while True:
            try:
                line = input('> ').strip()
            except (EOFError, KeyboardInterrupt):
                print('\nExiting console.'); break
            if not line: continue
            tokens = line.split()
            cmd = tokens[0].lower()
            if cmd=='exit':
                print('Terminating console...')
                if scheduler: scheduler.shutdown(); time.sleep(0.2)
                break
            if cmd=='help': print_help(); continue
            if cmd=='initialize':
                try:
                    scheduler = Scheduler('config.txt')
                    screen_manager = ScreenManager(scheduler)
                    initialized = True
                    print('initialize: configuration loaded and scheduler started.')
                    print(f" - num-cpu: {scheduler.config_num_cpu}")
                    print(f" - scheduler: {scheduler.config_scheduler}")
                    print(f" - quantum-cycles: {scheduler.config_quantum}")
                    print(f" - batch-process-freq: {scheduler.config_batch_freq} ticks")
                    print(f" - min-ins: {scheduler.config_min_ins} max-ins: {scheduler.config_max_ins}")
                    print(f" - delays-per-exec: {scheduler.config_delays} ticks")
                    print(f" - max-overall-mem: {scheduler.config_max_overall_mem} bytes")
                    print(f" - mem-per-frame: {scheduler.config_mem_per_frame} bytes")
                except Exception as e:
                    print('Error during initialize:', e)
                continue
            if not initialized:
                print("Error: Please run 'initialize' first (only 'initialize' and 'exit' available).")
                continue
            if cmd=='screen':
                screen_manager.handle_main(tokens)
            elif cmd=='scheduler-start':
                scheduler.start_batch_generation()
            elif cmd=='scheduler-stop':
                scheduler.stop_batch_generation()
            elif cmd=='report-util':
                scheduler.generate_util_report()
            elif cmd=='process-smi':
                if len(tokens)!=2:
                    print('Usage: process-smi <process_name>'); continue
                name = tokens[1]
                proc = scheduler.find_proc_by_name(name)
                if not proc:
                    print(f'Process {name} not found.'); continue
                info = proc.smi_summary()
                print(f"Process: {info['name']} (pid {info['pid']})")
                print(f"Status: {info['status']}")
                print(f"Memory base: 0x{info['mem_base']:x} size: {info['mem_size']} bytes")
                print(f"Instructions remaining: {info['instr_left']}")
                print('Variables:')
                for k,v in info['vars'].items(): print(f"  {k} = {v}")
            elif cmd=='vmstat':
                vmstat(scheduler)
            else:
                print('Unknown command. Type help to list commands.')
    finally:
        if scheduler: scheduler.shutdown()
        print('Console terminated.')

if __name__=='__main__':
    main()
