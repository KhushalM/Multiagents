from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List
import os
import dspy as dspy  # type: ignore
import wikipedia  # type: ignore

# ----------------------
# Load LM
# ----------------------

dspy.configure(lm=dspy.LM(model="gpt-5", api_key=os.getenv("OPENAI_API_KEY"), temperature=1.0))


# ----------------------
# DSPy Signature(s)
# ----------------------



class RetrieveWiki(dspy.Signature):
    """Retrieve and summarize relevant Wikipedia snippets for a user query."""

    query: str = dspy.InputField()
    passages: List[str] = dspy.OutputField(desc="A list of relevant snippets from Wikipedia.")


class AnswerWithCitations(dspy.Signature):
    """Answer the user's question using provided passages and include short citations."""

    question: str = dspy.InputField()
    passages: List[str] = dspy.InputField()
    answer: str = dspy.OutputField(desc="Concise answer with inline [n] citations referencing passages.")


# ----------------------
# Utilities
# ----------------------

def clean_text(text: str, max_len: int = 1200) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def search_wikipedia(query: str, top_k: int = 5) -> List[str]:
    """Search Wikipedia and return cleaned summaries/snippets of top results."""
    try:
        titles = wikipedia.search(query, results=top_k) or []
    except Exception:
        titles = []

    snippets: List[str] = []
    for i, title in enumerate(titles):
        try:
            page = wikipedia.page(title, auto_suggest=False)
            snippet = clean_text(page.summary or "")
            if snippet:
                snippets.append(f"[{i+1}] {title}: {snippet}")
        except Exception:
            continue
    return snippets


# ----------------------
# DSPy Modules
# ----------------------

class WikiRetriever(dspy.Module):
    """A simple retrieval module using Wikipedia as the corpus."""

    def __init__(self, top_k: int = 5):
        super().__init__()
        self.top_k = top_k
        self.retrieve = dspy.Predict(RetrieveWiki) 

    def forward(self, query: str) -> List[str]:
        passages = search_wikipedia(query, top_k=self.top_k)
        return passages


class WikiRAG(dspy.Module):
    """Minimal RAG pipeline: retrieve Wikipedia snippets, then answer with citations."""

    def __init__(self, top_k: int = 5):
        super().__init__()
        self.retriever = WikiRetriever(top_k=top_k)
        self.answerer = dspy.ChainOfThought(AnswerWithCitations)

    def forward(self, question: str) -> dspy.Prediction:
        passages = self.retriever(question)
        pred = self.answerer(question=question, passages=passages)
        print("Using Wiki RAG")
        return pred
    
def compile_rag(rag: WikiRAG) -> WikiRAG:
    """Few shot optimization for the WikiRAG module."""
    few = [
        dspy.Example(question="What is the capital of France?", passages=search_wikipedia("capital of France", top_k=3), answer="Paris is the capital of France. [1]").with_inputs("question"),
        dspy.Example(question="Who is the president of the United States?", passages=search_wikipedia("president of the United States", top_k=3), answer="Donald Trump is the president of the United States. [1]").with_inputs("question"),
        dspy.Example(question="Who is the current Manchester United manager?", passages=search_wikipedia("Manchester United manager", top_k=3), answer="Ruben Amorin is the current Manchester United manager. [1]").with_inputs("question"),
        dspy.Example(question="What happened on January 6th 2021?", passages=search_wikipedia("January 6th 2021", top_k=3), answer="The January 6th, 2021, attack on the United States Capitol was a riot by a mob of supporters of then-President Donald Trump, who stormed the Capitol in an attempt to overturn the 2020 presidential election results. [1]").with_inputs("question"),
        dspy.Example(question="Who is the CEO of Perplexity?", passages=search_wikipedia("Perplexity AI CEO", top_k=3), answer="Aravind Srinivasan is the CEO of Perplexity. [1]").with_inputs("question"),
        dspy.Example(question="When did Elon Musk buy Twitter?", passages=search_wikipedia("Elon Musk Twitter Purchase", top_k=3), answer="Elon Musk bought Twitter in 2022. [1]").with_inputs("question"),
    ]

    opt = dspy.BootstrapFewShot(
        metric = lambda gold, pred, trace: int(bool(pred.answer)),
        max_bootstrapped_demos=6,
        max_labeled_demos=6,
    )
    return opt.compile(rag, trainset=few)

RAG = compile_rag(WikiRAG(top_k=5))

def dspy_wiki_search(query: str) -> str:
    """Search Wikipedia"""
    passages = RAG.retriever(query)
    return "\n".join(f"- {p}" for p in passages) or "No passages found"

def dspy_wiki_rag(question: str, chat_history: str | None = None) -> str:
    """Answer a question using the WikiRAG pipeline. chat_history is optional."""
    response = RAG(question)
    return response.answer

# # ----------------------
# # Optimizer (optional in v1)
# # ----------------------

# @dataclass
# class FewShotExample:
#     question: str
#     passages: List[str]
#     answer: str


# def build_simple_optimizer(train_examples: List[FewShotExample]):
#     """Construct a small DSPy optimizer and return a compile function bound to the trainset."""
#     trainset = [
#         dspy.Example(question=ex.question, passages=ex.passages, answer=ex.answer).with_inputs("question", "passages")
#         for ex in train_examples
#     ]

#     optimizer = dspy.BootstrapFewShot(
#         metric=lambda gold, pred, trace: int(pred["answer"] is not None and len(pred["answer"]) > 0),
#         max_bootstrapped_demos=6,
#         max_labeled_demos=6,
#     )

#     def compile_program(program: dspy.Module) -> dspy.Module:
#         return optimizer.compile(program, trainset=trainset)

#     return compile_program


# # ----------------------
# # CLI Entrypoint
# # ----------------------

# def main():
#     import argparse
#     import os
#     import sys

#     parser = argparse.ArgumentParser(description="DSPy Wikipedia RAG agent")
#     parser.add_argument("question", type=str, nargs="*", help="Question to ask")
#     parser.add_argument("--top_k", type=int, default=5)
#     parser.add_argument("--model", type=str, default="gpt-5", help="Override LM model name")
#     parser.add_argument("--temperature", type=float, default=1.0)
#     parser.add_argument("--optimize", action="store_true", default=False, help="Run a tiny few-shot optimization demo")
#     parser.add_argument("--offline", action="store_true", help="Skip LM init; just show retrieved passages")
#     args = parser.parse_args()

#     if not args.offline:
#         try:
#             from .config import init_lm
#         except Exception:
#             # Fallback when run as a standalone script: add current directory to path
#             sys.path.append(os.path.dirname(__file__))
#             from config import init_lm
#         init_lm(model_name=args.model, temperature=args.temperature)

#     rag = Agent(top_k=args.top_k)

#     if args.optimize:
#         # Very small synthetic training example (optional)
#         demos = [
#             FewShotExample(
#                 question="Who discovered penicillin?",
#                 passages=search_wikipedia("penicillin discovery", top_k=3),
#                 answer="Alexander Fleming discovered penicillin in 1928. [1]",
#             )
#         ]


#         compile_fn = build_simple_optimizer(demos)
#         compiled_tool = compile_fn(WikiRAG(top_k=args.top_k))
#         # Re-wrap compiled retriever as a Tool for ReAct
#         def _impl(query: str) -> dspy.Prediction:
#             print(f"[Tool] wikipedia_search (compiled) called query=\"{query}\"")
#             passages = compiled_tool.retriever(query)
#             return dspy.Prediction(passages=passages)
#         rag.wiki_tool = dspy.Tool(
#             func=_impl,
#             name="wikipedia_search",
#             desc="Search Wikipedia (compiled retriever).",
#             args={"query": None},
#             arg_types={"query": str},
#             arg_desc={"query": "User query to search on Wikipedia"},
#         )
#         rag.reasoner = dspy.ReAct(AgentSignature, tools=[rag.wiki_tool])

#     if args.question:
#         question = " ".join(args.question).strip()
#     else:
#         question = input("Enter a question: ").strip() or "What is the capital of France?"

#     if args.offline:
#         passages = search_wikipedia(question, top_k=args.top_k)
#         print("\nRetrieved passages:")
#         for p in passages:
#             print("- " + p)
#         return
#     pred = rag(question)
#     print("\nAnswer:\n" + pred.answer)
#     #print(f"History: {dspy.inspect_history(n=1)}")


# if __name__ == "__main__":
#     main()


