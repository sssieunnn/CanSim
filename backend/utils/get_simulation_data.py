import os

def get_simulation_data(results, best_drug, detailed=False):
    if not results:
        raise ValueError("results가 비어 있습니다. 시뮬레이션 결과를 확인하세요.")
    num_weeks = len(results[0]["images"])
    week_interval = 1 if detailed else 4
    week_list = list(range(0, num_weeks, week_interval))

    os.makedirs("temp_images", exist_ok=True)
    data = []
    for result in results:
        drug_data = {
            "drug": result["drug"],
            "is_best": result["drug"] == best_drug["drug"],
            "weeks": []
        }
        for w in week_list:
            image_path = f"temp_images/{result['drug'].replace(' ', '_')}_week{w+1}.png"
            result["images"][w].save(image_path)
            drug_data["weeks"].append({
                "week": w+1,
                "score": float(result["scores"][w]),
                "image_path": image_path
            })
        data.append(drug_data)
    return data