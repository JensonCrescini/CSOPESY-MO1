// main.cpp
// CSOPESY OS Emulator - single-file C++ translation of provided Python emulator
// Compile: g++ -std=c++20 main.cpp -pthread -O2 -o os_emulator

#include <bits/stdc++.h>
using namespace std;

constexpr double TICK_DURATION_SEC = 0.05; // 50 ms per tick
constexpr unsigned int UINT16_MAX_VAL = 65535u;

struct Instr {
    string type; // PRINT, DECLARE, ADD, SUBTRACT, SLEEP, FOR
    vector<string> args; // for PRINT/DECLARE/ADD/SUBTRACT/SLEEP
    vector<Instr> body; // for FOR
    int repeats = 0;    // for FOR
};

// Utility
static unsigned int clamp_uint16(int v) {
    if (v < 0) return 0;
    if (v > (int)UINT16_MAX_VAL) return UINT16_MAX_VAL;
    return (unsigned int)v;
}

// Forward
class Scheduler;

class Process {
public:
    static atomic<unsigned long> global_id_counter;
    unsigned long pid;
    string name;
    vector<Instr> instructions;
    // stack frames: pair of (instr_list pointer, index, repeats)
    struct Frame { vector<Instr>* list_ptr; size_t idx; unsigned int repeats; };
    vector<Frame> pc_stack;
    unordered_map<string,unsigned int> vars;
    bool finished = false;
    vector<string> logs;
    chrono::steady_clock::time_point create_time;
    chrono::steady_clock::time_point start_time;
    chrono::steady_clock::time_point end_time;
    unsigned long total_cpu_ticks = 0;
    int delays_per_exec = 0; // ticks per instruction busy

    Process(const string& name_, int instr_count, int delays_per_exec_, mt19937 &rng) :
        name(name_), delays_per_exec(delays_per_exec_)
    {
        pid = ++global_id_counter;
        create_time = chrono::steady_clock::now();
        instructions = generate_instructions(instr_count, rng);
        // push top frame
        pc_stack.push_back({&instructions, 0, 1});
    }

    // Generate randomized instructions similar to Python version
    vector<Instr> generate_instructions(int n, mt19937 &rng) {
        vector<Instr> ins;
        uniform_real_distribution<> d01(0.0,1.0);
        uniform_int_distribution<> varnum(0,9);
        uniform_int_distribution<> val100(0,100);
        uniform_int_distribution<> smallval(0,50);
        uniform_int_distribution<> sleepdur(1,3);
        for (int i=0;i<n;i++) {
            // Weighted choices: PRINT 0.25, ADD 0.2, SUB 0.15, SLEEP 0.15, DECLARE 0.15, FOR 0.1
            double x = d01(rng);
            if (x < 0.25) {
                Instr it; it.type="PRINT"; it.args.push_back("Hello world from " + name + "!");
                ins.push_back(move(it));
            } else if (x < 0.45) { // ADD
                Instr it; it.type="ADD";
                string v1 = "v" + to_string(varnum(rng));
                string v2 = "v" + to_string(varnum(rng));
                if (d01(rng) < 0.4) {
                    int val = smallval(rng);
                    it.args = {v1, v2, to_string(val)};
                } else {
                    string v3 = "v" + to_string(varnum(rng));
                    it.args = {v1, v2, v3};
                }
                ins.push_back(move(it));
            } else if (x < 0.60) { // SUBTRACT
                Instr it; it.type="SUBTRACT";
                string v1 = "v" + to_string(varnum(rng));
                string v2 = "v" + to_string(varnum(rng));
                if (d01(rng) < 0.4) {
                    int val = smallval(rng);
                    it.args = {v1, v2, to_string(val)};
                } else {
                    string v3 = "v" + to_string(varnum(rng));
                    it.args = {v1, v2, v3};
                }
                ins.push_back(move(it));
            } else if (x < 0.75) { // SLEEP
                Instr it; it.type="SLEEP";
                it.args.push_back(to_string(sleepdur(rng)));
                ins.push_back(move(it));
            } else if (x < 0.90) { // DECLARE
                Instr it; it.type="DECLARE";
                string var = "v" + to_string(varnum(rng));
                int val = val100(rng);
                it.args = {var, to_string(val)};
                ins.push_back(move(it));
            } else { // FOR
                Instr it; it.type="FOR";
                uniform_int_distribution<> repeats(2,4);
                uniform_int_distribution<> bodylen(1,3);
                it.repeats = repeats(rng);
                int bl = bodylen(rng);
                for (int j=0;j<bl;j++) {
                    double r = d01(rng);
                    if (r < 0.45) {
                        Instr p; p.type="PRINT"; p.args.push_back("Loop msg from " + name);
                        it.body.push_back(move(p));
                    } else if (r < 0.77) {
                        Instr a; a.type="ADD"; a.args = {"v" + to_string(varnum(rng)), "v" + to_string(varnum(rng)), "1"};
                        it.body.push_back(move(a));
                    } else {
                        Instr s; s.type="SLEEP"; s.args.push_back("1");
                        it.body.push_back(move(s));
                    }
                }
                ins.push_back(move(it));
            }
        }
        return ins;
    }

    // approximate remaining instruction count
    unsigned long instructions_remaining_count() {
        unsigned long total = 0;
        for (auto &f : pc_stack) {
            long left = (long)f.list_ptr->size() - (long)f.idx;
            if (left > 0) total += (unsigned long)left * (unsigned long)f.repeats;
        }
        return total;
    }

    // peek next instruction (not advancing)
    optional<Instr> peek_next_instruction() {
        if (finished) return {};
        if (pc_stack.empty()) return {};
        auto &f = pc_stack.back();
        if (f.idx >= f.list_ptr->size()) return {};
        return (*f.list_ptr)[f.idx];
    }

    // Execute exactly one logical instruction.
    // Returns: ticks (positive busy ticks), negative ticks for SLEEP (relinquish), 0 for finished/no-op
    // outputs: printed messages
    pair<int, vector<string>> step_instruction() {
        vector<string> outputs;
        if (finished) return {0, outputs};
        while (!pc_stack.empty()) {
            Frame &frame = pc_stack.back();
            if (frame.idx >= frame.list_ptr->size()) {
                // finished frame iteration
                if (frame.repeats > 1) {
                    frame.repeats -= 1;
                    frame.idx = 0;
                    // continue with new iteration
                } else {
                    // pop frame
                    pc_stack.pop_back();
                }
                continue;
            }
            Instr instr = (*frame.list_ptr)[frame.idx];
            // advance index now
            frame.idx += 1;
            int ticks = max(1, delays_per_exec); // busy ticks at least delays_per_exec
            if (instr.type == "PRINT") {
                string msg = instr.args.size() ? instr.args[0] : "";
                outputs.push_back(msg);
            } else if (instr.type == "DECLARE") {
                if (instr.args.size() >= 2) {
                    string var = instr.args[0];
                    int val = stoi(instr.args[1]);
                    vars[var] = clamp_uint16(val);
                }
            } else if (instr.type == "ADD") {
                if (instr.args.size() >= 3) {
                    string var1 = instr.args[0];
                    string op2 = instr.args[1];
                    string op3 = instr.args[2];
                    int v2 = 0, v3 = 0;
                    if (!op2.empty() && isalpha(op2[0])) {
                        v2 = vars.count(op2) ? (int)vars[op2] : 0;
                    } else v2 = stoi(op2);
                    if (!op3.empty() && isalpha(op3[0])) {
                        v3 = vars.count(op3) ? (int)vars[op3] : 0;
                    } else v3 = stoi(op3);
                    int res = v2 + v3;
                    vars[var1] = clamp_uint16(res);
                }
            } else if (instr.type == "SUBTRACT") {
                if (instr.args.size() >= 3) {
                    string var1 = instr.args[0];
                    string op2 = instr.args[1];
                    string op3 = instr.args[2];
                    int v2 = 0, v3 = 0;
                    if (!op2.empty() && isalpha(op2[0])) {
                        v2 = vars.count(op2) ? (int)vars[op2] : 0;
                    } else v2 = stoi(op2);
                    if (!op3.empty() && isalpha(op3[0])) {
                        v3 = vars.count(op3) ? (int)vars[op3] : 0;
                    } else v3 = stoi(op3);
                    int res = v2 - v3;
                    vars[var1] = clamp_uint16(res);
                }
            } else if (instr.type == "SLEEP") {
                int dur = 1;
                if (instr.args.size()) dur = stoi(instr.args[0]);
                // return negative to signal relinquish for dur ticks
                return {-dur, outputs};
            } else if (instr.type == "FOR") {
                // push body frame
                if (instr.body.size() > 0 && instr.repeats > 0) {
                    // We need to store a copy of body in heap so frame can reference it.
                    // Simplify by storing body vector in a holder owned by this process.
                    // We'll push body onto instr_frames vector and reference it.
                    // For simplicity inside single-file: we copy the body into instructions_storage and reference it.
                    instructions_storage.push_back(instr.body);
                    Frame nf;
                    nf.list_ptr = &instructions_storage.back();
                    nf.idx = 0;
                    nf.repeats = instr.repeats;
                    pc_stack.push_back(nf);
                    // executing FOR itself costs ticks (already set)
                }
            } else {
                outputs.push_back("[" + name + "] Unknown instruction " + instr.type);
            }
            // After execution, check empty stack => finished
            if (pc_stack.empty()) {
                finished = true;
                end_time = chrono::steady_clock::now();
            }
            return {ticks, outputs};
        }
        // stack empty
        finished = true;
        end_time = chrono::steady_clock::now();
        return {0, {}};
    }

    // SMI summary (similar to Python)
    struct SmiSummary {
        string name;
        unsigned long pid;
        string status;
        unsigned long instr_left;
        unordered_map<string,unsigned int> vars_snapshot;
        vector<string> logs_tail;
        unsigned long total_cpu_ticks;
    };

    SmiSummary smi_summary() {
        SmiSummary s;
        s.name = name;
        s.pid = pid;
        s.status = finished ? "Finished" : "Running";
        s.instr_left = instructions_remaining_count();
        s.vars_snapshot = vars;
        size_t tail = min<size_t>(logs.size(), 10);
        s.logs_tail.assign(logs.end()-tail, logs.end());
        s.total_cpu_ticks = total_cpu_ticks;
        return s;
    }

private:
    // storage for dynamically added FOR bodies
    vector<vector<Instr>> instructions_storage;
};

atomic<unsigned long> Process::global_id_counter(0);

// Scheduler
class Scheduler {
public:
    Scheduler(const string& config_path) : rng(random_device{}()) {
        load_config(config_path);
        ready_queue = deque<shared_ptr<Process>>();
        running_procs.clear();
        finished_procs.clear();
        cores.assign(config_num_cpu, 0);
        core_busy_ticks.assign(config_num_cpu, 0);
        shutdown_flag = false;
        ticker_running = false;
        proc_gen_running = false;

        // spawn worker threads for cores
        for (int cid=0; cid<config_num_cpu; ++cid) {
            worker_threads.emplace_back(&Scheduler::cpu_worker_loop, this, cid);
        }

        start_ticker();
    }

    ~Scheduler() {
        shutdown();
        for (auto &t : worker_threads) {
            if (t.joinable()) t.join();
        }
        if (ticker_thread.joinable()) ticker_thread.join();
        if (proc_gen_thread.joinable()) proc_gen_thread.join();
    }

    // config.txt parser (space-separated key value)
    void load_config(const string& path) {
        // set defaults
        config_num_cpu = stoi(DEFAULT_CONFIG.at("num-cpu"));
        config_scheduler = DEFAULT_CONFIG.at("scheduler");
        config_quantum = stoi(DEFAULT_CONFIG.at("quantum-cycles"));
        config_batch_freq = stoi(DEFAULT_CONFIG.at("batch-process-freq"));
        config_min_ins = stoi(DEFAULT_CONFIG.at("min-ins"));
        config_max_ins = stoi(DEFAULT_CONFIG.at("max-ins"));
        config_delays = stoi(DEFAULT_CONFIG.at("delays-per-exec"));

        ifstream f(path);
        if (!f) {
            cout << "[initialize] Warning: " << path << " not found. Using defaults.\n";
        } else {
            string key, val;
            while (f >> key >> val) {
                if (key == "num-cpu") config_num_cpu = stoi(val);
                else if (key == "scheduler") {
                    config_scheduler = val;
                    for (auto &c: config_scheduler) c = tolower(c);
                }
                else if (key == "quantum-cycles") config_quantum = stoi(val);
                else if (key == "batch-process-freq") config_batch_freq = stoi(val);
                else if (key == "min-ins") config_min_ins = stoi(val);
                else if (key == "max-ins") config_max_ins = stoi(val);
                else if (key == "delays-per-exec") config_delays = stoi(val);
            }
        }
        if (config_num_cpu < 1) config_num_cpu = 1;
        if (config_num_cpu > 128) config_num_cpu = 128;
        // resize structures possibly used before constructing threads
        // (we set cores/core_busy_ticks later)
    }

    // start ticker thread
    void start_ticker() {
        if (ticker_running) return;
        ticker_running = true;
        ticker_thread = thread(&Scheduler::_ticker_loop, this);
    }

    // proc generation
    void start_batch_generation() {
        lock_guard<mutex> lg(mu);
        if (proc_gen_running) {
            cout << "scheduler already generating processes.\n";
            return;
        }
        proc_gen_running = true;
        proc_gen_thread = thread(&Scheduler::_proc_gen_loop, this);
        cout << "scheduler-start: batch generation started.\n";
    }

    void stop_batch_generation() {
        {
            lock_guard<mutex> lg(mu);
            if (!proc_gen_running) {
                cout << "scheduler-stop: no batch generation running.\n";
                return;
            }
            proc_gen_running = false;
            cout << "scheduler-stop: batch generation stopping...\n";
        }
        if (proc_gen_thread.joinable()) proc_gen_thread.join();
        cout << "scheduler-stop: batch generation stopped.\n";
    }

    // list processes (returns copy vectors)
    pair<vector<shared_ptr<Process>>, vector<shared_ptr<Process>>> list_processes() {
        lock_guard<mutex> lg(mu);
        vector<shared_ptr<Process>> running;
        for (auto &p : ready_queue) running.push_back(p);
        for (auto &kv : running_procs) running.push_back(kv.second);
        vector<shared_ptr<Process>> finished;
        for (auto &kv : finished_procs) finished.push_back(kv.second);
        return {running, finished};
    }

    shared_ptr<Process> find_proc_by_name(const string& name) {
        lock_guard<mutex> lg(mu);
        for (auto &p : ready_queue) if (p->name == name && !p->finished) return p;
        for (auto &kv : running_procs) if (kv.second->name == name && !kv.second->finished) return kv.second;
        // per spec: finished should not be accessible via -r
        return nullptr;
    }

    // generate utilization report and write csopesy-log.txt
    struct UtilReport {
        double utilization_pct;
        int cores_used;
        int cores_available;
        vector<string> running;
        vector<string> finished;
        vector<unordered_map<string,string>> per_process;
    };

    UtilReport generate_util_report() {
        lock_guard<mutex> lg(mu);
        unsigned long total_time_ticks = max<unsigned long>(1, total_ticks);
        unsigned long total_core_ticks = 0;
        for (auto v : core_busy_ticks) total_core_ticks += v;
        unsigned long max_possible_ticks = total_time_ticks * (unsigned long)config_num_cpu;
        double utilization_pct = max_possible_ticks>0 ? (double)total_core_ticks / (double)max_possible_ticks * 100.0 : 0.0;
        int cores_used = 0;
        for (auto c : cores) if (c != 0) cores_used++;
        int cores_available = config_num_cpu - cores_used;
        vector<string> running_names;
        for (auto &p : ready_queue) running_names.push_back(p->name);
        for (auto &kv : running_procs) running_names.push_back(kv.second->name);
        vector<string> finished_names;
        for (auto &kv : finished_procs) finished_names.push_back(kv.second->name);

        // per process summary
        unordered_map<unsigned long, shared_ptr<Process>> all_known;
        for (auto &p : ready_queue) all_known[p->pid] = p;
        for (auto &kv : running_procs) all_known[kv.first] = kv.second;
        for (auto &kv : finished_procs) all_known[kv.first] = kv.second;

        vector<unordered_map<string,string>> per_proc_summary;
        for (auto &kv : all_known) {
            auto p = kv.second;
            unordered_map<string,string> s;
            s["name"] = p->name;
            s["pid"] = to_string(p->pid);
            s["status"] = p->finished ? "Finished" : "Running";
            s["instr_left"] = to_string(p->instructions_remaining_count());
            s["total_cpu_ticks"] = to_string(p->total_cpu_ticks);
            per_proc_summary.push_back(s);
        }

        // write to csopesy-log.txt
        ofstream out("csopesy-log.txt");
        out << "=== CSOPESY CPU UTILIZATION REPORT ===\n";
        out << "Timestamp: " << chrono::system_clock::to_time_t(chrono::system_clock::now()) << "\n";
        out << "Total ticks elapsed: " << total_time_ticks << "\n";
        out << "CPU cores configured: " << config_num_cpu << "\n";
        out << fixed << setprecision(2);
        out << "CPU utilization: " << utilization_pct << "% (" << total_core_ticks << " busy ticks of " << max_possible_ticks << " possible)\n";
        out << "Cores used (now): " << cores_used << "\n";
        out << "Cores available (now): " << cores_available << "\n\n";
        out << "Running processes:\n";
        for (auto &r : running_names) out << " - " << r << "\n";
        out << "\nFinished processes:\n";
        for (auto &f : finished_names) out << " - " << f << "\n";
        out << "\nPer-process summary:\n";
        for (auto &s : per_proc_summary) {
            out << " - " << s.at("name") << " (pid " << s.at("pid") << "): " << s.at("status")
                << ", instr_left=" << s.at("instr_left") << ", cpu_ticks=" << s.at("total_cpu_ticks") << "\n";
        }
        out.close();
        cout << "report-util: csopesy-log.txt generated.\n";

        UtilReport rep;
        rep.utilization_pct = utilization_pct;
        rep.cores_used = cores_used;
        rep.cores_available = cores_available;
        rep.running = running_names;
        rep.finished = finished_names;
        rep.per_process = per_proc_summary;
        return rep;
    }

    void shutdown() {
        {
            lock_guard<mutex> lg(mu);
            shutdown_flag = true;
            proc_gen_running = false;
            ticker_running = false;
        }
        // notify threads (workers check flag)
        if (proc_gen_thread.joinable()) proc_gen_thread.join();
        // workers joined in destructor
        cout << "Shutting down scheduler...\n";
    }

    // Create a process and push to ready queue (used by screen -s)
    shared_ptr<Process> create_process_with_name(const string& pname) {
        lock_guard<mutex> lg(mu);
        int ins = uniform_int_distribution<int>(config_min_ins, config_max_ins)(rng);
        auto p = make_shared<Process>(pname, ins, config_delays, rng);
        ready_queue.push_back(p);
        return p;
    }

private:
    // Default config
    const unordered_map<string,string> DEFAULT_CONFIG {
        {"num-cpu","2"},
        {"scheduler","fcfs"},
        {"quantum-cycles","3"},
        {"batch-process-freq","5"},
        {"min-ins","3"},
        {"max-ins","8"},
        {"delays-per-exec","0"}
    };

    // Configs
    int config_num_cpu = 2;
    string config_scheduler = "fcfs"; // fcfs or rr
    int config_quantum = 3;
    int config_batch_freq = 5;
    int config_min_ins = 3;
    int config_max_ins = 8;
    int config_delays = 0;

    // runtime state
    deque<shared_ptr<Process>> ready_queue;
    unordered_map<unsigned long, shared_ptr<Process>> running_procs;
    unordered_map<unsigned long, shared_ptr<Process>> finished_procs;

    vector<unsigned long> cores; // pid assigned or 0
    vector<unsigned long> core_busy_ticks;

    // synchronization
    mutex mu;

    vector<thread> worker_threads;
    thread ticker_thread;
    thread proc_gen_thread;
    bool ticker_running;
    bool proc_gen_running;
    bool shutdown_flag;
    unsigned long total_ticks = 0;

    // for rng and process generation
    mt19937 rng;

    // worker helpers
    vector<unsigned long> core_busy_ticks_snapshot; // unused

    // worker loop
    void cpu_worker_loop(int core_id) {
        // each worker repeatedly tries to fetch process and run according to scheduling
        while (true) {
            {
                lock_guard<mutex> lg(mu);
                if (shutdown_flag) break;
            }
            shared_ptr<Process> proc = nullptr;
            {
                lock_guard<mutex> lg(mu);
                if (!ready_queue.empty()) {
                    proc = ready_queue.front();
                    ready_queue.pop_front();
                    running_procs[proc->pid] = proc;
                    if ((int)cores.size() > core_id) cores[core_id] = proc->pid;
                } else {
                    if ((int)cores.size() > core_id) cores[core_id] = 0;
                }
            }
            if (!proc) {
                this_thread::sleep_for(chrono::duration<double>(TICK_DURATION_SEC/4.0));
                continue;
            }
            if (proc->start_time.time_since_epoch().count() == 0) proc->start_time = chrono::steady_clock::now();

            if (config_scheduler == "fcfs") {
                while (!proc->finished) {
                    {
                        lock_guard<mutex> lg(mu);
                        if (shutdown_flag) break;
                    }
                    auto [ticks, outputs] = proc->step_instruction();
                    if (ticks == 0) break;
                    if (ticks < 0) {
                        int sleep_ticks = -ticks;
                        {
                            lock_guard<mutex> lg(mu);
                            if (cores.size() > (size_t)core_id) cores[core_id] = 0;
                            running_procs.erase(proc->pid);
                        }
                        // sleep real time without busy counting
                        this_thread::sleep_for(chrono::duration<double>(sleep_ticks * TICK_DURATION_SEC));
                        {
                            lock_guard<mutex> lg(mu);
                            if (!proc->finished) ready_queue.push_back(proc);
                        }
                        break;
                    }
                    // busy-wait simulation: occupy core for 'ticks' ticks and increment counters
                    for (int tt=0; tt<ticks; ++tt) {
                        {
                            lock_guard<mutex> lg(mu);
                            core_busy_ticks[core_id] += 1;
                            proc->total_cpu_ticks += 1;
                        }
                        this_thread::sleep_for(chrono::duration<double>(TICK_DURATION_SEC));
                    }
                    for (auto &ln : outputs) proc->logs.push_back(ln);
                }
                // move to finished if done
                if (proc->finished) {
                    lock_guard<mutex> lg(mu);
                    if (cores.size() > (size_t)core_id) cores[core_id] = 0;
                    running_procs.erase(proc->pid);
                    finished_procs[proc->pid] = proc;
                }
            } else { // round robin
                int remaining_quantum = config_quantum;
                bool preempted = false;
                while (!proc->finished && remaining_quantum > 0) {
                    {
                        lock_guard<mutex> lg(mu);
                        if (shutdown_flag) break;
                    }
                    auto [ticks, outputs] = proc->step_instruction();
                    if (ticks == 0) break;
                    if (ticks < 0) {
                        int sleep_ticks = -ticks;
                        {
                            lock_guard<mutex> lg(mu);
                            if (cores.size() > (size_t)core_id) cores[core_id] = 0;
                            running_procs.erase(proc->pid);
                        }
                        this_thread::sleep_for(chrono::duration<double>(sleep_ticks * TICK_DURATION_SEC));
                        {
                            lock_guard<mutex> lg(mu);
                            if (!proc->finished) ready_queue.push_back(proc);
                        }
                        preempted = true;
                        break;
                    }
                    int ticks_to_consume = min(ticks, remaining_quantum);
                    for (int tt=0; tt<ticks_to_consume; ++tt) {
                        {
                            lock_guard<mutex> lg(mu);
                            core_busy_ticks[core_id] += 1;
                            proc->total_cpu_ticks += 1;
                            remaining_quantum -= 1;
                        }
                        this_thread::sleep_for(chrono::duration<double>(TICK_DURATION_SEC));
                    }
                    if (ticks > ticks_to_consume) {
                        // treat as partially executed: requeue at front to resume soon
                        lock_guard<mutex> lg(mu);
                        ready_queue.push_front(proc);
                        preempted = true;
                        break;
                    }
                    for (auto &ln : outputs) proc->logs.push_back(ln);
                } // end rr while
                if (proc->finished) {
                    lock_guard<mutex> lg(mu);
                    if (cores.size() > (size_t)core_id) cores[core_id] = 0;
                    running_procs.erase(proc->pid);
                    finished_procs[proc->pid] = proc;
                } else if (!preempted) {
                    lock_guard<mutex> lg(mu);
                    if (cores.size() > (size_t)core_id) cores[core_id] = 0;
                    running_procs.erase(proc->pid);
                    ready_queue.push_back(proc);
                }
            }
        } // outer while
    }

    // ticker increments total_ticks at TICK_DURATION
    void _ticker_loop() {
        while (true) {
            {
                lock_guard<mutex> lg(mu);
                if (shutdown_flag) break;
                if (!ticker_running) break;
            }
            this_thread::sleep_for(chrono::duration<double>(TICK_DURATION_SEC));
            lock_guard<mutex> lg(mu);
            total_ticks += 1;
        }
    }

    // batch generation
    void _proc_gen_loop() {
        int counter = 1;
        long long last_tick = -1;
        while (true) {
            {
                lock_guard<mutex> lg(mu);
                if (shutdown_flag) break;
                if (!proc_gen_running) break;
            }
            this_thread::sleep_for(chrono::duration<double>(TICK_DURATION_SEC / 2.0));
            {
                lock_guard<mutex> lg(mu);
                if (!proc_gen_running) break;
                unsigned long current_tick = total_ticks;
                if (last_tick == -1) last_tick = (long long)current_tick;
                if ((long long)(current_tick - last_tick) >= config_batch_freq) {
                    // create process
                    char buf[32];
                    snprintf(buf, sizeof(buf), "p%03d", counter);
                    int ins = uniform_int_distribution<int>(config_min_ins, config_max_ins)(rng);
                    auto p = make_shared<Process>(string(buf), ins, config_delays, rng);
                    ready_queue.push_back(p);
                    counter++;
                    last_tick = current_tick;
                }
            }
        }
    }
};

// Screen Manager / CLI wrapper
class ScreenManager {
public:
    ScreenManager(Scheduler* s) : sched(s) {}

    void handle_main(const vector<string>& tokens) {
        if (tokens.size() < 2) {
            cout << "Usage: screen -s <name> | screen -ls | screen -r <name>\n";
            return;
        }
        string flag = tokens[1];
        if (flag == "-s") {
            if (tokens.size() != 3) {
                cout << "Usage: screen -s <process_name>\n";
                return;
            }
            string name = tokens[2];
            if (_name_exists(name)) {
                cout << "Process " << name << " already exists (running or finished). Choose another name.\n";
                return;
            }
            // create and append to ready queue
            auto p = sched->create_process_with_name(name);
            cout << "Created and attached to process " << name << ".\n";
            _process_screen_loop(name);
        } else if (flag == "-ls") {
            auto [running, finished] = sched->list_processes();
            cout << "=== screen -ls ===\n";
            // cores used and available
            {
                // We don't have direct public cores, so emulate by calling report (cheap)
            }
            // print running
            cout << "\nRunning processes:\n";
            for (auto &p : running) {
                cout << " - " << p->name << " (pid " << p->pid << ") " << (p->finished ? "[Finished]" : "") << "\n";
            }
            cout << "\nFinished processes:\n";
            for (auto &p : finished) {
                cout << " - " << p->name << " (pid " << p->pid << ")\n";
            }
        } else if (flag == "-r") {
            if (tokens.size() != 3) {
                cout << "Usage: screen -r <process_name>\n";
                return;
            }
            string name = tokens[2];
            auto p = sched->find_proc_by_name(name);
            if (!p) {
                cout << "Process " << name << " not found.\n";
                return;
            }
            cout << "Re-attaching to process " << name << ".\n";
            _process_screen_loop(name);
        } else {
            cout << "Invalid screen flags. Use -s, -ls, or -r.\n";
        }
    }

private:
    Scheduler* sched;

    bool _name_exists(const string& name) {
        auto [running, finished] = sched->list_processes();
        for (auto &p : running) if (p->name == name) return true;
        for (auto &p : finished) if (p->name == name) return true;
        return false;
    }

    void _process_screen_loop(const string& name) {
        while (true) {
            cout << "(" << name << ")> " << flush;
            string cmd;
            if (!getline(cin, cmd)) {
                cout << "\nReturning to main menu.\n";
                break;
            }
            if (cmd.size() == 0) continue;
            if (cmd == "exit") {
                cout << "Detaching and returning to main menu.\n";
                break;
            } else if (cmd == "process-smi") {
                // find process in ready/running/finished
                shared_ptr<Process> proc = nullptr;
                {
                    auto [running, finished] = sched->list_processes();
                    for (auto &p : running) if (p->name == name) { proc = p; break; }
                    if (!proc) {
                        // check running procs via find_proc_by_name (only checks ready+running)
                        proc = sched->find_proc_by_name(name);
                    }
                    if (!proc) {
                        // check finished by generating report and scanning file list (inelegant)
                        // But Scheduler::list_processes returns finished processes too, so we retrieved finished earlier
                        for (auto &p : finished) if (p->name == name) { proc = p; break; }
                    }
                }
                if (!proc) {
                    cout << "Process " << name << " not found.\n";
                    continue;
                }
                auto info = proc->smi_summary();
                cout << "Process: " << info.name << " (pid " << info.pid << ")\n";
                if (proc->finished) cout << "Finished!\n";
                cout << "Status: " << info.status << "\n";
                cout << "Instructions remaining (approx): " << info.instr_left << "\n";
                cout << "Total CPU ticks consumed: " << info.total_cpu_ticks << "\n";
                cout << "Variables:\n";
                for (auto &kv : info.vars_snapshot) {
                    cout << "  " << kv.first << " = " << kv.second << "\n";
                }
                cout << "Logs (last up to 10):\n";
                for (auto &ln : info.logs_tail) cout << "  " << ln << "\n";
            } else {
                cout << "Unknown screen command. Supported: process-smi, exit\n";
            }
        }
    }
};


// Main CLI
void print_help() {
    cout << R"(
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
)";
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    Scheduler* scheduler = nullptr;
    ScreenManager* screen_manager = nullptr;
    bool initialized = false;

    cout << "Welcome to CSOPESY OS Emulator (C++). Type 'help' for commands.\n";
    try {
        while (true) {
            cout << "> " << flush;
            string line;
            if (!getline(cin, line)) {
                cout << "\nExiting console.\n";
                break;
            }
            // trim
            auto ltrim = [](string &s){ s.erase(s.begin(), find_if(s.begin(), s.end(), [](int ch){ return !isspace(ch); })); };
            auto rtrim = [](string &s){ s.erase(find_if(s.rbegin(), s.rend(), [](int ch){ return !isspace(ch); }).base(), s.end()); };
            ltrim(line); rtrim(line);
            if (line.empty()) continue;
            // split tokens
            vector<string> tokens;
            {
                istringstream iss(line);
                string t;
                while (iss >> t) tokens.push_back(t);
            }
            string cmd = tokens[0];
            for (auto &c : cmd) c = tolower(c);

            if (cmd == "exit") {
                cout << "Terminating console...\n";
                if (scheduler) {
                    scheduler->shutdown();
                    this_thread::sleep_for(chrono::milliseconds(200));
                    delete scheduler; scheduler = nullptr;
                }
                break;
            }
            if (cmd == "help") {
                print_help();
                continue;
            }
            if (cmd == "initialize") {
                if (initialized) {
                    cout << "Already initialized.\n";
                    continue;
                }
                try {
                    scheduler = new Scheduler("config.txt");
                    screen_manager = new ScreenManager(scheduler);
                    initialized = true;
                    cout << "initialize: configuration loaded and scheduler started.\n";
                    // Print config summary by re-reading file (Scheduler holds defaults only private)
                    // For simplicity, print minimal info: num-cpu unknown externally, we assume 2 default
                    cout << " - Scheduler started. Use 'screen -ls' and other commands now.\n";
                } catch (const exception &e) {
                    cout << "Error during initialize: " << e.what() << "\n";
                }
                continue;
            }
            if (!initialized) {
                cout << "Error: Please run 'initialize' first (only 'initialize' and 'exit' available).\n";
                continue;
            }

            if (cmd == "screen") {
                screen_manager->handle_main(tokens);
            } else if (cmd == "scheduler-start") {
                scheduler->start_batch_generation();
            } else if (cmd == "scheduler-stop") {
                scheduler->stop_batch_generation();
            } else if (cmd == "report-util") {
                scheduler->generate_util_report();
            } else {
                cout << "Unknown command. Type 'help' to list commands.\n";
            }
        }
    } catch (const exception &e) {
        cerr << "Fatal exception: " << e.what() << "\n";
    }

    if (scheduler) {
        scheduler->shutdown();
        delete scheduler;
    }
    cout << "Console terminated.\n";
    return 0;
}
