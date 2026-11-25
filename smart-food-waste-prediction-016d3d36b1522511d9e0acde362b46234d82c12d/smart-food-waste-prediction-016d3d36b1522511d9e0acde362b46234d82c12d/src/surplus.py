
def estimate_surplus(predicted_demand, planned_inventory):

    # Calculate the number of potentially unused meals
    surplus = max(0, planned_inventory - predicted_demand)

    # Calculate surplus percentage
    if planned_inventory > 0:
        surplus_percentage = (surplus / planned_inventory) * 100
    else:
        surplus_percentage = 0

    # Classify surplus risk
    if surplus_percentage < 10:
        risk = "Low"
        recommendation = "Inventory levels look reasonable."

    elif surplus_percentage < 25:
        risk = "Medium"
        recommendation = "Consider reducing the next preparation quantity."

    else:
        risk = "High"
        recommendation = (
            "Reduce preparation, consider discounts, "
            "or redistribute safe surplus food."
        )

    return {
        "predicted_demand": predicted_demand,
        "planned_inventory": planned_inventory,
        "potential_surplus": surplus,
        "surplus_percentage": round(surplus_percentage, 2),
        "risk_level": risk,
        "recommendation": recommendation
    }


# Example execution
if __name__ == "__main__":

    result = estimate_surplus(
        predicted_demand=805,
        planned_inventory=1000
    )

    print("\nFOOD SURPLUS ESTIMATION")
    print("----------------------")

    for key, value in result.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
