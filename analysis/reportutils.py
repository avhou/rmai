import statutils as ut

def compare(model_A: str, model_B: str, labelled_error_rates: dict) -> None:
    model_A_predictions = labelled_error_rates[model_A]
    model_B_predictions = labelled_error_rates[model_B]
    # ----------------
    # Mean error rates
    # ----------------
    mean_error_rate_car = ut.mean_error_rate(model_A_predictions, "car")
    mean_error_rate_bus = ut.mean_error_rate(model_A_predictions, "bus")
    print(f"MODEL {model_A}")
    print(f"\tMean error rate for car: {round(mean_error_rate_car, 3)}")
    print(f"\tMean error rate for bus: {round(mean_error_rate_bus, 3)}")
    mean_error_rate_car = ut.mean_error_rate(model_B_predictions, "car")
    mean_error_rate_bus = ut.mean_error_rate(model_B_predictions, "bus")
    print(f"MODEL {model_B}")
    print(f"\tMean error rate for car: {round(mean_error_rate_car, 3)}")
    print(f"\tMean error rate for bus: {round(mean_error_rate_bus, 3)}")
    # -----------------------------------
    # Paired sample t-test on error rates
    # -----------------------------------
    ttest_car = ut.paired_samples_ttest(model_A_predictions, model_B_predictions, "car")
    ttest_bus = ut.paired_samples_ttest(model_A_predictions, model_B_predictions, "bus")
    print(f"\nH_a: error rates for model {model_B} < error rates for model {model_A}")
    print(f"Paired sample t-test (two-tailed)")
    print(f"\t car t-statistic: {round(ttest_car.statistic, 3)}")
    print(f"\t car p-value: {round(ttest_car.pvalue, 3)}")
    print(f"\t bus t-statistic: {round(ttest_bus.statistic, 3)}")
    print(f"\t bus p-value: {round(ttest_bus.pvalue, 3)}")
    print(f"Paired sample t-test for car (one-tailed)")
    ttest_alt_car = ut.t_test(model_A_predictions, model_B_predictions, "car", "greater")
    ttest_alt_bus = ut.t_test(model_A_predictions, model_B_predictions, "bus", "greater")
    print(f"\t car p-value: {round(ttest_alt_car, 3)}")
    print(f"\t bus p-value: {round(ttest_alt_bus, 3)}")
    # ---------------------------
    # McNemar test on error rates
    # ---------------------------
    mcnemar_car = ut.mc_nemar_test(model_A_predictions, model_B_predictions, "car")
    mcnemar_bus = ut.mc_nemar_test(model_A_predictions, model_B_predictions, "bus")
    print(f"Mc Nemar for car")
    print(round(mcnemar_car.pvalue, 3))
    print(f"Mc Nemar for bus")
    print(round(mcnemar_bus.pvalue, 3))