# logic_engine.py
class KnowledgeBase:
    """Propositional Knowledge Base: facts (percepts) + Horn-clause rules, with forward chaining."""

    def __init__(self):
        self.facts = set()    # unique fact strings, e.g. 'TargetVisible'
        self.rules = []       # tuples: (premise_list, conclusion_string)

    def tell_fact(self, fact_string):
        self.facts.add(fact_string)

    def tell_rule(self, premise_list, conclusion_string):
        self.rules.append((list(premise_list), conclusion_string))

    def clear_facts(self):
        self.facts.clear()

    def forward_chain(self):
        """Data-driven inference: keep firing rules until a full pass adds no new fact."""
        new_facts_added = True
        while new_facts_added:
            new_facts_added = False
            for premises, conclusion in self.rules:
                if conclusion not in self.facts:
                    # Modus Ponens: all premises known => conclusion is entailed
                    if all(p in self.facts for p in premises):
                        self.facts.add(conclusion)
                        new_facts_added = True   # a new fact may unlock other rules
