"""ZeroDesperdício — interface Streamlit (MVP de demonstração).

Rodar:  make app   (ou  .venv/bin/streamlit run app/streamlit_app.py)
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import altair as alt
import pandas as pd
import streamlit as st

from zerodesperdicio import config
from zerodesperdicio.agents import DataNotReady, Request
from zerodesperdicio.app_support import RESTART_NOTE, build_orchestrator, panel_eval, production_message
from zerodesperdicio.index import read_manifest
from zerodesperdicio.ingredient_rules import STAPLES
from zerodesperdicio.pantry import ExpiryStatus, PantryItem, PantryState, PantryStore, assess
from zerodesperdicio.textutil import md_escape

CHART_COLOR = "#3D8B5E"  # validado (dataviz) contra superfície clara e escura
STATUS_ICON = {
    ExpiryStatus.EXPIRED: "⛔",
    ExpiryStatus.TODAY: "🔴",
    ExpiryStatus.SOON: "🟠",
    ExpiryStatus.OK: "🟢",
    ExpiryStatus.UNKNOWN: "⚪",
}
RESTRICTION_OPTIONS = ["lactose", "glúten", "frutos do mar", "carne", "amendoim", "ovo"]

st.set_page_config(page_title="ZeroDesperdício", page_icon="🥕", layout="wide")


@st.cache_resource(show_spinner="Carregando base de receitas e modelo de embeddings…")
def get_orchestrator(root: str):
    return build_orchestrator()


def example_items(today: date) -> list[PantryItem]:
    return [
        PantryItem("ovos", "6 unidades", today),
        PantryItem("espinafre", "1 maço", today + timedelta(days=1)),
        PantryItem("tomate", "3 unidades", today + timedelta(days=2)),
        PantryItem("banana", "5 unidades", today + timedelta(days=3)),
        PantryItem("leite", "1 litro", today + timedelta(days=6)),
        PantryItem("queijo", "200 g"),
        PantryItem("iogurte", "1 pote", today - timedelta(days=1)),
    ]


# ------------------------------------------------------------------ estado
store = PantryStore(config.pantry_file())
if "pantry" not in st.session_state:
    state, warning = store.load()
    st.session_state.pantry = state
    st.session_state.pantry_warning = warning
pantry: PantryState = st.session_state.pantry


def save():
    store.save(st.session_state.pantry)


st.title("🥕 ZeroDesperdício")
st.caption("Assistente de IA para aproveitar os alimentos que já estão em casa — Challenge 2 · Qualidade de Vida")

try:
    orch, prod_info = get_orchestrator(str(config.root()))
except DataNotReady as exc:
    st.error(f"**Dados ainda não preparados.** {exc}")
    st.markdown("Rode estes comandos na pasta do projeto e recarregue a página:")
    st.code("make setup   # instala as dependências (uv)\n"
            ".venv/bin/python -I -m zerodesperdicio.pipeline fetch   # baixa o Wikilivros (~5 min)\n"
            ".venv/bin/python -I -m zerodesperdicio.pipeline build   # gera a base de receitas\n"
            ".venv/bin/python -I -m zerodesperdicio.index build      # gera o índice vetorial",
            language="bash")
    st.stop()

data_manifest = json.loads(config.manifest_path().read_text(encoding="utf-8"))
index_manifest = read_manifest(config.index_dir()) or {}

# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.subheader("Estado do sistema")
    st.markdown(f"📚 **Base:** {len(orch.recipes):,} receitas".replace(",", "."))
    st.caption(f"{data_manifest['source']} · {data_manifest['license']}")
    st.markdown(f"🧭 **Índice:** {index_manifest.get('count', 0):,} vetores".replace(",", "."))
    st.caption(index_manifest.get("model", ""))
    if orch.llm is not None:
        st.markdown(f"🤖 **LLM:** ligado ({orch.llm.name})")
    else:
        st.markdown("🤖 **LLM:** desligado — explicação por regras")
        st.caption("Para ligar: copie `.env.example` para `.env` e preencha `ANTHROPIC_API_KEY`.")
    st.divider()
    top_n = st.slider("Quantas sugestões", 3, 10, 5)
    assume_staples = st.checkbox("Considerar que tenho água, sal, açúcar, óleo e pimenta", value=True)
    st.divider()
    st.caption("**Limites:** o app não diz se um alimento está próprio para consumo; datas e prioridades "
               "são regras fixas; as receitas vêm do Wikilivros e podem ter erros; não substitui "
               "orientação de nutricionista.")

tab_main, tab_cmp, tab_data = st.tabs(["Despensa e sugestões", "LLM × RAG", "Dados e métricas"])

# ------------------------------------------------------------------ aba 1
with tab_main:
    if st.session_state.get("pantry_warning"):
        st.warning(st.session_state.pantry_warning)
    today = date.today()
    left, right = st.columns([5, 7], gap="large")

    with left:
        st.subheader("Minha despensa")
        with st.form("add_item", clear_on_submit=True):
            c1, c2 = st.columns([3, 2])
            name = c1.text_input("Alimento", placeholder="ex.: tomate")
            qty = c2.text_input("Quantidade", placeholder="ex.: 3 unidades")
            exp = st.date_input("Validade (deixe vazio se não souber)", value=None, format="DD/MM/YYYY")
            if st.form_submit_button("Adicionar", use_container_width=True):
                if name.strip():
                    pantry.items.append(PantryItem(name.strip(), qty.strip(), exp))
                    save()
                else:
                    st.warning("Escreva o nome do alimento.")
        b1, b2 = st.columns(2)
        if b1.button("Carregar exemplo", key="load_example", use_container_width=True):
            pantry.items = example_items(today)
            save()
        if b2.button("Limpar despensa", key="clear", use_container_width=True):
            pantry.items = []
            save()

        entries = assess(pantry.items, today)
        if not entries:
            st.info("A despensa está vazia. Adicione alimentos ou clique em **Carregar exemplo**.")
        for i, e in enumerate(entries):
            row = st.columns([6, 2, 2])
            when = e.status.value if e.days_left is None or e.days_left <= 0 else f"vence em {e.days_left} dias"
            qty_txt = f" · {md_escape(e.item.quantity)}" if e.item.quantity else ""
            row[0].markdown(f"{STATUS_ICON[e.status]} **{md_escape(e.item.name)}**{qty_txt}  \n"
                            f"<small>{when}</small>", unsafe_allow_html=True)
            idx = pantry.items.index(e.item)
            if row[1].button("Usei", key=f"used_{i}", use_container_width=True):
                pantry.items.pop(idx)
                pantry.used += 1
                save()
                st.rerun()
            if row[2].button("Descartei", key=f"disc_{i}", use_container_width=True):
                pantry.items.pop(idx)
                pantry.discarded += 1
                save()
                st.rerun()
        expired = [e for e in entries if e.status == ExpiryStatus.EXPIRED]
        if expired:
            st.warning("⛔ Vencidos (ficam fora das sugestões): "
                       + ", ".join(md_escape(e.item.name) for e in expired))
        if pantry.used or pantry.discarded:
            st.caption(f"Histórico: {pantry.used} aproveitados · {pantry.discarded} descartados")

    with right:
        st.subheader("O que fazer com isso?")
        pref = st.text_input("Vontade ou ocasião (opcional)", placeholder="ex.: sobremesa, algo rápido para o jantar")
        c1, c2 = st.columns([3, 2])
        chosen = c1.multiselect("Restrições", RESTRICTION_OPTIONS)
        extra = c2.text_input("Outras restrições", placeholder="separe por vírgula")
        restrictions = chosen + [r.strip() for r in extra.split(",") if r.strip()]
        if st.button("Sugerir receitas", key="suggest", type="primary", use_container_width=True):
            with st.spinner("Os agentes estão trabalhando…"):
                st.session_state.result = orch.handle(Request(
                    items=list(pantry.items), today=today, restrictions=restrictions,
                    preference=pref, assume_staples=assume_staples, top_n=top_n))

        res = st.session_state.get("result")
        if res is not None:
            if res.unknown:
                st.info("Sem receitas na base para: " + ", ".join(md_escape(u) for u in res.unknown))
            if res.message and not res.suggestions:
                st.warning(res.message)
            else:
                badge = "🤖 redigida pelo LLM" if res.explanation.source == "llm" else "📏 explicação por regras"
                with st.container(border=True):
                    st.caption(badge)
                    st.markdown(res.explanation.text)
                    if res.explanation.error:
                        st.caption(f"⚠️ {res.explanation.error}")
                if res.removed_by_restrictions:
                    st.caption(f"{res.removed_by_restrictions} receita(s) candidatas removidas pelas restrições.")
                for n, s in enumerate(res.suggestions, start=1):
                    r = s.recipe
                    with st.container(border=True):
                        st.markdown(f"**[R{n}] {md_escape(r.title)}**  ·  pontuação {s.score:.2f}")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Você já tem", f"{s.coverage:.0%}")
                        m2.metric("Urgência aproveitada", f"{s.urgency_share:.0%}")
                        m3.metric("Similaridade", f"{s.semantic:.2f}")
                        used = ", ".join(f"{STATUS_ICON[e.status]} {md_escape(e.item.name)}" for e in s.used)
                        st.markdown(f"**Usa da despensa:** {used}")
                        if s.staples:
                            st.markdown("**Itens básicos:** " + ", ".join(md_escape(x) for x in s.staples))
                        st.markdown("**Falta:** " + (", ".join(md_escape(x) for x in s.missing) or "nada 🎉"))
                        with st.expander("Ingredientes e modo de preparo"):
                            st.markdown("\n".join(f"- {md_escape(x)}" for x in r.ingredients))
                            st.markdown("\n".join(f"{i}. {md_escape(x)}" for i, x in enumerate(r.steps, 1)))
                        st.markdown(f"<small>Fonte: [{md_escape(r.source)}]({r.url}) · revisão de "
                                    f"{r.revision_date} · {r.license}</small>", unsafe_allow_html=True)
                with st.expander("Como os agentes chegaram aqui (rastro)"):
                    st.dataframe(pd.DataFrame([{"agente": t.agent, "ação": t.action, "detalhe": t.detail,
                                                "ms": round(t.ms, 1)} for t in res.trace]),
                                 hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ aba 2
with tab_cmp:
    st.subheader("LLM sozinho × recuperação (RAG)")
    st.markdown("Mesma despensa, duas abordagens. Medimos **quanto de cada receita a pessoa não tem** "
                "(fora da despensa e dos itens básicos) e se a receita tem **fonte verificável**.")
    if orch.llm is None:
        st.info("A comparação precisa do LLM ligado (veja a barra lateral). As sugestões do RAG continuam "
                "funcionando sem ele.")
    elif not pantry.items:
        st.info("Cadastre itens na despensa primeiro.")
    elif st.button("Comparar agora", key="compare"):
        from zerodesperdicio.evaluation import llm_only_recipes, outside_pantry_rate
        from zerodesperdicio.llm import LLMError

        entries = [e for e in assess(pantry.items, date.today()) if e.status != ExpiryStatus.EXPIRED]
        keys = {e.key for e in entries}
        staples = STAPLES if assume_staples else frozenset()
        rows = []
        try:
            for rec in llm_only_recipes(orch.llm, [e.item.name for e in entries]):
                rows.append({"abordagem": "LLM sozinho", "receita": rec["titulo"],
                             "fora da despensa": outside_pantry_rate(rec["ingredientes"], keys, staples),
                             "fonte": "—"})
        except LLMError as exc:
            st.error(str(exc))
        except Exception as exc:  # o LLM nunca derruba a tela
            st.error(f"Falha inesperada do LLM: {exc.__class__.__name__}")
        res = orch.handle(Request(items=list(pantry.items), today=date.today(), top_n=3,
                                  assume_staples=assume_staples))
        for s in res.suggestions:
            rows.append({"abordagem": "RAG", "receita": s.recipe.title,
                         "fora da despensa": outside_pantry_rate(s.recipe.ingredient_names, keys, staples),
                         "fonte": s.recipe.url})
        df = pd.DataFrame(rows)
        if not df.empty:
            st.dataframe(df, hide_index=True, use_container_width=True,
                         column_config={"fora da despensa": st.column_config.NumberColumn(format="percent"),
                                        "fonte": st.column_config.LinkColumn()})
            summary = df.groupby("abordagem")["fora da despensa"].mean().reset_index()
            st.dataframe(summary, hide_index=True,
                         column_config={"fora da despensa": st.column_config.NumberColumn(
                             "média fora da despensa", format="percent")})

# ------------------------------------------------------------------ aba 3
with tab_data:
    s = data_manifest["stats"]
    st.subheader("Pipeline de dados")
    c = st.columns(4)
    c[0].metric("Páginas lidas", f"{s['pages']:,}".replace(",", "."))
    c[1].metric("Receitas na base", f"{s['recipes']:,}".replace(",", "."))
    c[2].metric("Ingredientes distintos", f"{s['unique_keys']:,}".replace(",", "."))
    c[3].metric("Linhas sem ingrediente reconhecido",
                f"{s['lines_without_key'] / max(s['ingredient_lines'], 1):.1%}")
    funnel = pd.DataFrame([
        {"etapa": "1. variações de receita extraídas", "receitas": s["parsed"]},
        {"etapa": "2. sem bebidas alcoólicas", "receitas": s["parsed"] - s["dropped_alcohol"]},
        {"etapa": "3. com ≥ 2 ingredientes e preparo",
         "receitas": s["parsed"] - s["dropped_alcohol"] - s["dropped_few_ingredients"] - s["dropped_no_steps"]},
        {"etapa": "4. sem duplicatas (base final)", "receitas": s["recipes"]},
    ])
    base = alt.Chart(funnel).encode(
        y=alt.Y("etapa:N", sort=None, title=None),
        x=alt.X("receitas:Q", title="receitas"),
        tooltip=["etapa", "receitas"])
    st.altair_chart(base.mark_bar(color=CHART_COLOR, cornerRadiusEnd=4, height=22)
                    + base.mark_text(align="left", dx=4).encode(text="receitas:Q"),
                    use_container_width=True)
    st.caption(f"Fonte: {data_manifest['source']} ({data_manifest['license']}). "
               f"{data_manifest['policy'].capitalize()}. Gerado em {data_manifest['built_at']}.")
    if "manual_sample_accuracy" in data_manifest:
        m = data_manifest["manual_sample_accuracy"]
        st.caption(f"Conferência manual da normalização: {m['correct']}/{m['n']} linhas corretas "
                   f"(amostra aleatória, semente {m['seed']}).")

    st.subheader("Ingredientes mais frequentes")
    top = pd.DataFrame(s["top_keys"][:20], columns=["ingrediente", "receitas"])
    chart = alt.Chart(top).encode(
        y=alt.Y("ingrediente:N", sort="-x", title=None),
        x=alt.X("receitas:Q", title="receitas que usam"),
        tooltip=["ingrediente", "receitas"])
    st.altair_chart(chart.mark_bar(color=CHART_COLOR, cornerRadiusEnd=4, height=14), use_container_width=True)

    st.subheader("Versão em produção")
    kind, text = production_message(prod_info)
    getattr(st, kind)(text)
    st.caption(RESTART_NOTE)

    st.subheader("Qualidade da recuperação")
    ev, origem = panel_eval(prod_info)
    if ev is not None:
        names = {"semantic": "só semântica (embeddings)", "lexical": "só ingredientes (índice invertido)",
                 "hybrid": "híbrida + ranking (usada no app)"}
        st.dataframe(pd.DataFrame([{"modo": names[k], **ev[k]} for k in ("semantic", "lexical", "hybrid")]),
                     hide_index=True, use_container_width=True)
        st.caption(f"{origem} Teste 'known item': {ev['n']} receitas sorteadas (semente {ev['seed']}); a despensa "
                   f"recebe {ev['keep']:.0%} dos ingredientes de cada uma e medimos se a própria receita volta "
                   "entre as k primeiras. MRR = média de 1/posição.")
    else:
        st.info(f"{origem} Rode `make eval` para medir a recuperação.")
