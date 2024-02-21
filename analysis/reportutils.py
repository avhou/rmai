import errorratestats as ut
import ioustats as iot

def compare_error_rates(model_A: str, model_B: str, labelled_error_rates: dict) -> None:
    with open(f"error_rates_{model_A}_{model_B}.txt", "w") as report_file:
        model_A_predictions = labelled_error_rates[model_A]
        model_B_predictions = labelled_error_rates[model_B]
        # ----------------
        # Mean error rates
        # ----------------
        mean_error_rate_car = ut.mean_error_rate(model_A_predictions, "car")
        mean_error_rate_bus = ut.mean_error_rate(model_A_predictions, "bus")
        report_file.write(f"MODEL {model_A} \n")
        report_file.write(f"\tMean error rate for car: {round(mean_error_rate_car, 3)} \n")
        report_file.write(f"\tMean error rate for bus: {round(mean_error_rate_bus, 3)} \n")
        mean_error_rate_car = ut.mean_error_rate(model_B_predictions, "car")
        mean_error_rate_bus = ut.mean_error_rate(model_B_predictions, "bus")
        report_file.write(f"MODEL {model_B} \n")
        report_file.write(f"\tMean error rate for car: {round(mean_error_rate_car, 3)} \n")
        report_file.write(f"\tMean error rate for bus: {round(mean_error_rate_bus, 3)} \n")
        # -----------------------------------
        # Paired sample t-test on error rates
        # -----------------------------------
        ttest_car = ut.paired_samples_ttest(model_A_predictions, model_B_predictions, "car")
        ttest_bus = ut.paired_samples_ttest(model_A_predictions, model_B_predictions, "bus")
        report_file.write(f"\nH_a: mean(error rate) for model {model_B} != mean(error rate) for model {model_A} \n")
        report_file.write(f"Paired sample t-test (two-tailed) \n")
        report_file.write(f"\t car t-statistic: {round(ttest_car.statistic, 3)} \n")
        report_file.write(f"\t car p-value: {round(ttest_car.pvalue, 3)} \n")
        report_file.write(f"\t bus t-statistic: {round(ttest_bus.statistic, 3)} \n")
        report_file.write(f"\t bus p-value: {round(ttest_bus.pvalue, 3)} \n")
        report_file.write(f"\nH_a: mean(error rate) for model {model_B} < mean(error rate) for model {model_A} \n")
        report_file.write(f"Paired sample t-test (one-tailed) \n")
        ttest_alt_car = ut.t_test(model_A_predictions, model_B_predictions, "car", "greater")
        ttest_alt_bus = ut.t_test(model_A_predictions, model_B_predictions, "bus", "greater")
        report_file.write(f"\t car p-value: {round(ttest_alt_car, 3)} \n")
        report_file.write(f"\t bus p-value: {round(ttest_alt_bus, 3)} \n")
        # ---------------------------
        # McNemar test on error rates
        # ---------------------------
        mcnemar_car = ut.mc_nemar_test(model_A_predictions, model_B_predictions, "car")
        mcnemar_bus = ut.mc_nemar_test(model_A_predictions, model_B_predictions, "bus")
        report_file.write("\n")
        report_file.write(f"Mc Nemar for car \n")
        report_file.write(f"\t {round(mcnemar_car.pvalue, 3)} \n")
        report_file.write(f"Mc Nemar for bus \n")
        report_file.write(f"\t {round(mcnemar_bus.pvalue, 3)} \n")

def compare_ious(model_A: str, model_B: str, labelled_ious: dict) -> None:
    with open(f"ious_{model_A}_{model_B}.txt", "w") as report_file:
        model_A_ious = labelled_ious[model_A]
        model_B_ious = labelled_ious[model_B]
        # -----------------------------
        # Paired sample t-test on IoU's
        # -----------------------------
        ttest_car = iot.paired_samples_ttest(model_A_ious, model_B_ious, "car")
        ttest_bus = iot.paired_samples_ttest(model_A_ious, model_B_ious, "bus")
        report_file.write(f"\nH_a: mean(IoU) for model {model_B} != mean(IoU) for model {model_A} \n")
        report_file.write(f"Paired sample t-test (two-tailed) \n")
        report_file.write(f"\t car t-statistic: {round(ttest_car.statistic, 3)} \n")
        report_file.write(f"\t car p-value: {round(ttest_car.pvalue, 3)} \n")
        report_file.write(f"\t bus t-statistic: {round(ttest_bus.statistic, 3)} \n")
        report_file.write(f"\t bus p-value: {round(ttest_bus.pvalue, 3)} \n")