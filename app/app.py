import streamlit as st
import threading
import time

from core.ttp_loader import load_instance
from solvers.aco.aco_solver import ACOSolver
from solvers.gwo.gwo_solver import GWOSolver
from solvers.ga import GA
from solvers.sa import SA
from visuals.convergence import plot_convergence, plot_convergence_comparison
from visuals.tour_map import plot_tour
from visuals.animations import animate_tour, plot_pheromone, animate_pheromone, animate_wolves

INSTANCES = {
    "eil51 — uncorrelated": "benchmarks/eil51/eil51_n50_uncorr_01.ttp",
    "eil51 — bounded-strongly-corr": "benchmarks/eil51/eil51_n50_bounded-strongly-corr_01.ttp",
}

SOLVERS = {
    "ACO": lambda: ACOSolver(max_iterations=500),
    "GWO": lambda: GWOSolver(max_iterations=500),
    "GA": lambda: GA(max_iterations=500),
    "SA": lambda: SA(),
}


class _Stopped(Exception):
    pass


def _solve_all(instance, n_runs, algo_name, out, stop_event):
    results, solvers = [], []
    for run_i in range(n_runs):
        if stop_event.is_set():
            break
        solver = SOLVERS[algo_name]()

        def on_iter(it, score, _i=run_i, _s=solver):
            out["progress"] = (_i + it / _s.max_iterations) / n_runs
            if stop_event.is_set():
                raise _Stopped()

        solver.on_iteration = on_iter
        try:
            result = solver.solve(instance)
        except _Stopped:
            if solver._best_solution is not None:
                result = solver._finalize()
                result.runtime = time.time() - getattr(solver, "_start_time", time.time())
                results.append(result)
                solvers.append(solver)
            break

        results.append(result)
        solvers.append(solver)

    out["results"] = results
    out["solvers"] = solvers
    out["done"] = True


st.set_page_config(page_title="TTP Metaheuristics", layout="wide")
st.title("Traveling Thief Problem — Metaheuristics")

with st.sidebar:
    inst_name = st.selectbox("Instance", list(INSTANCES.keys()))
    algo_name = st.selectbox("Algorithm", list(SOLVERS.keys()))
    n_runs = st.slider("Runs", 1, 10, 3)
    run_btn = st.button("Run", use_container_width=True)

if run_btn:
    instance = load_instance(INSTANCES[inst_name])
    out = {"progress": 0.0, "done": False}
    stop_event = threading.Event()
    st.session_state["out"] = out
    st.session_state["stop_event"] = stop_event
    st.session_state["running"] = True
    st.session_state["instance"] = instance
    st.session_state["inst_name"] = inst_name
    st.session_state["algo_name"] = algo_name
    threading.Thread(
        target=_solve_all,
        args=(instance, n_runs, algo_name, out, stop_event),
        daemon=True,
    ).start()
    st.rerun()

if st.session_state.get("running", False):
    out = st.session_state["out"]
    st.progress(out["progress"])
    if st.button("Stop", type="secondary", use_container_width=True):
        st.session_state["stop_event"].set()
    if not out["done"]:
        time.sleep(0.3)
        st.rerun()
    else:
        st.session_state["running"] = False
        results = out.get("results", [])
        solvers = out.get("solvers", [])
        if results:
            best_idx = max(range(len(results)), key=lambda i: results[i].convergence[-1])
            st.session_state["results"] = results
            st.session_state["best_solver"] = solvers[best_idx]
            st.session_state["best_result"] = results[best_idx]
        st.rerun()

if "results" in st.session_state and not st.session_state.get("running", False):
    results = st.session_state["results"]
    best_solver = st.session_state["best_solver"]
    best_result = st.session_state["best_result"]
    instance = st.session_state["instance"]
    inst_name = st.session_state["inst_name"]
    algo_name = st.session_state["algo_name"]

    scores = [r.convergence[-1] for r in results]
    col1, col2, col3 = st.columns(3)
    col1.metric("Best", f"{max(scores):.1f}")
    col2.metric("Worst", f"{min(scores):.1f}")
    col3.metric("Mean", f"{sum(scores) / len(scores):.1f}")

    tab1, tab2, tab3, tab4 = st.tabs(["All runs", "Mean ± std", "Best tour", "Algorithm visual"])

    with tab1:
        st.pyplot(plot_convergence(results, title=f"{algo_name} on {inst_name}"))

    with tab2:
        st.pyplot(plot_convergence_comparison(results, title=f"{algo_name} on {inst_name}"))

    with tab3:
        st.pyplot(plot_tour(instance, best_result, title=f"Best tour — {algo_name}"))
        anim = animate_tour(instance, best_result)
        anim.save("results/tour_anim.gif", writer="pillow", fps=15)
        st.image("results/tour_anim.gif", caption="Tour drawn edge by edge")

    with tab4:
        st.write("algo:", algo_name, "| solver:", type(best_solver).__name__,
                 "| has scores_history:", hasattr(best_solver, "scores_history"),
                 "| len:", len(best_solver.scores_history) if hasattr(best_solver, "scores_history") else "n/a")
        if algo_name == "ACO" and best_solver.tau_history:
            st.pyplot(plot_pheromone(best_solver.tau, title="Final pheromone matrix"))
            anim = animate_pheromone(best_solver.tau_history)
            anim.save("results/pheromone_anim.gif", writer="pillow", fps=3)
            st.image("results/pheromone_anim.gif", caption="Pheromone at each improvement")
        elif algo_name == "GWO" and hasattr(best_solver, "scores_history") and best_solver.scores_history:
            anim = animate_wolves(best_solver.scores_history)
            anim.save("results/wolves_anim.gif", writer="pillow", fps=10)
            st.image("results/wolves_anim.gif", caption="Wolf population converging toward alpha")
        else:
            st.info("No algorithm-specific visual for this solver.")
