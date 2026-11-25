
from predict import predict_demand
from surplus import estimate_surplus


def main():
    print("\nSMART FOOD WASTE PREDICTION SYSTEM")
    print("=" * 40)

    # Step 1: Predict food demand
    predicted_demand = predict_demand(
        week=150,
        center_id=10,
        meal_id=1885,
        checkout_price=150.0,
        base_price=160.0,
        emailer_for_promotion=1,
        homepage_featured=0,
        category="Beverages",
        cuisine="Thai",
        city_code=590,
        region_code=56,
        center_type="TYPE_A",
        op_area=2.5
    )

    print(f"\nPredicted Demand: {predicted_demand} orders")

    # Step 2: Ask for planned inventory
    try:
        planned_inventory = int(
            input("Enter planned inventory (number of meals): ")
        )

        if planned_inventory < 0:
            raise ValueError

    except ValueError:
        print("Please enter a valid non-negative whole number.")
        return

    # Step 3: Estimate potential surplus
    result = estimate_surplus(
        predicted_demand,
        planned_inventory
    )

    # Step 4: Display the final report
    print("\nFOOD SURPLUS REPORT")
    print("-" * 40)

    print(f"Predicted Demand: {result['predicted_demand']}")
    print(f"Planned Inventory: {result['planned_inventory']}")
    print(f"Potential Surplus: {result['potential_surplus']}")
    print(f"Surplus Percentage: {result['surplus_percentage']}%")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Recommendation: {result['recommendation']}")


if __name__ == "__main__":
    main()
