"""用云端 qwen3.8-27b 独立做一遍模板里的题，与模板 reference_answer 比对，输出分歧清单。

用法: .venv/bin/python tools/crosscheck_answers.py [paper_id ...]
输出: data/templates/{paper_id}/crosscheck.json（只含分歧题）
"""
import asyncio, json, os, re, sys

from openai import AsyncOpenAI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TPL = os.path.join(ROOT, "data", "templates")


def load_env():
    env = {}
    for line in open(os.path.join(ROOT, ".env")):
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.strip().split("=", 1)
            env[k] = v
    return env


PROMPT = """你是中国初三{subject}老师。独立解答下面这道题，不要参考任何给定答案。
最后一行严格输出：答案：<最终答案>（选择题只写字母；填空写最简结果；解答题写各小问最终结果，用；分隔）。

{context}题号 {number}（{qtype}）
{stem}
{options}"""


def norm(s):
    s = str(s or "").strip().lower()
    return re.sub(r"[\s$\\{}，,。.；;：:（）()]", "", s)


async def solve(client, model, sem, paper, q, context):
    opts = "\n".join(q.get("options") or [])
    msg = PROMPT.format(subject="数学" if paper["subject"] == "math" else "英语", number=q["number"],
                        qtype=q["type"], stem=q["stem"], options=opts,
                        context=f"【材料】\n{context}\n\n" if context else "")
    async with sem:
        r = await client.chat.completions.create(model=model, messages=[{"role": "user", "content": msg}],
                                                 temperature=0, max_tokens=1500,
                                                 extra_body={"enable_thinking": False})
    text = r.choices[0].message.content or ""
    m = re.findall(r"答案[:：]\s*(.+)", text)
    return text, (m[-1].strip() if m else text.strip().splitlines()[-1])


async def run(pid, client, model):
    path = os.path.join(TPL, pid, "template.json")
    t = json.load(open(path))
    sem = asyncio.Semaphore(8)
    jobs = []
    for p in t["pages"]:
        for q in p["questions"]:
            if q["type"] in ("listening", "writing"):
                continue
            ctx = t.get("passages", {}).get(q.get("passage_id") or "", "")
            jobs.append((q, solve(client, model, sem, t, q, ctx)))
    res = await asyncio.gather(*[j for _, j in jobs], return_exceptions=True)
    diffs = []
    for (q, _), r in zip(jobs, res):
        if isinstance(r, Exception):
            diffs.append({"id": q["id"], "number": q["number"], "error": str(r)}); continue
        full, ans = r
        ref = q.get("reference_answer")
        refs = ref if isinstance(ref, list) else [ref]
        if q["type"] == "choice":
            ok = norm(ans)[:1] == norm(refs[0])[:1]
        else:
            ok = any(norm(x) and (norm(x) in norm(ans) or norm(ans) in norm(x)) for x in refs)
        if not ok:
            diffs.append({"id": q["id"], "number": q["number"], "type": q["type"], "template": ref,
                          "qwen": ans, "qwen_full": full[-800:]})
    json.dump(diffs, open(os.path.join(TPL, pid, "crosscheck.json"), "w"), ensure_ascii=False, indent=1)
    print(f"{pid}: {len(jobs)} 题，分歧 {len(diffs)}：", [d["number"] for d in diffs])


async def main():
    env = load_env()
    client = AsyncOpenAI(base_url=env["CLOUD_LLM_BASE_URL"], api_key=env["CLOUD_LLM_API_KEY"], timeout=120)
    pids = sys.argv[1:] or sorted(d for d in os.listdir(TPL) if os.path.exists(os.path.join(TPL, d, "template.json")))
    for pid in pids:
        await run(pid, client, env["CLOUD_LLM_MODEL"])


if __name__ == "__main__":
    asyncio.run(main())
