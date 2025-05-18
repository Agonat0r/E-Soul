from dataclasses import dataclass
from typing import List, Dict, Set
import random
from datetime import datetime

@dataclass
class SoulTrait:
    name: str
    value: float  # 0.0 to 1.0
    description: str
    category: str  # e.g., "emotional", "spiritual", "karmic"
    created_at: datetime = datetime.now()

class Theory:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.required_traits: Dict[str, float] = {}  # trait_name -> expected_value
        self.confidence: float = 1.0  # How confident we are in this theory
        self.last_updated = datetime.now()

    def matches_traits(self, traits: List[SoulTrait]) -> bool:
        for trait_name, expected_value in self.required_traits.items():
            matching_trait = next((t for t in traits if t.name == trait_name), None)
            if not matching_trait:
                return False
            if abs(matching_trait.value - expected_value) > 0.2:  # 20% tolerance
                return False
        return True

    def adjust_for_traits(self, traits: List[SoulTrait]):
        for trait in traits:
            if trait.name in self.required_traits:
                # Gradually adjust theory to match observed traits
                current = self.required_traits[trait.name]
                self.required_traits[trait.name] = current * 0.9 + trait.value * 0.1
        self.last_updated = datetime.now()

class SoulSimulation:
    def __init__(self):
        self.traits: List[SoulTrait] = []
        self.theories: List[Theory] = []
        self.trait_categories = ["emotional", "spiritual", "karmic", "consciousness"]
        self.initialize_theories()

    def initialize_theories(self):
        # Add some initial theories
        theory1 = Theory("Balance Theory", "A soul in perfect balance maintains equilibrium between all aspects")
        theory1.required_traits = {
            "emotional_stability": 0.5,
            "spiritual_awareness": 0.5,
            "karmic_balance": 0.5
        }
        self.theories.append(theory1)

    def generate_new_trait(self) -> SoulTrait:
        # Generate a new random trait
        category = random.choice(self.trait_categories)
        name = f"{category}_trait_{len(self.traits)}"
        value = random.random()
        description = f"Generated {category} trait with value {value:.2f}"
        return SoulTrait(name, value, description, category)

    def evolve_traits(self):
        # Randomly modify existing traits
        for trait in self.traits:
            if random.random() < 0.3:  # 30% chance to evolve each trait
                trait.value = max(0.0, min(1.0, trait.value + random.uniform(-0.1, 0.1)))

    def match_theories(self):
        for theory in self.theories:
            if not theory.matches_traits(self.traits):
                theory.adjust_for_traits(self.traits)
                theory.confidence *= 0.95  # Reduce confidence when theory doesn't match

    def run_simulation_step(self):
        # Generate new trait
        new_trait = self.generate_new_trait()
        self.traits.append(new_trait)
        
        # Evolve existing traits
        self.evolve_traits()
        
        # Match and adjust theories
        self.match_theories()

    def get_simulation_state(self) -> Dict:
        return {
            "traits": self.traits,
            "theories": self.theories,
            "trait_count": len(self.traits),
            "theory_count": len(self.theories)
        }

# Example usage
if __name__ == "__main__":
    simulation = SoulSimulation()
    
    # Run simulation for 10 steps
    for _ in range(10):
        simulation.run_simulation_step()
        state = simulation.get_simulation_state()
        print(f"\nStep {_ + 1}:")
        print(f"Traits: {len(state['traits'])}")
        print(f"Theories: {len(state['theories'])}")
        print("Latest trait:", state['traits'][-1]) 