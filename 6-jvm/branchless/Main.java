public class Main {
    private static final int[] branchlessArray = { 1, 2, 3, 4 };

    private static int branchlessCalculation(int i) {
        return branchlessArray[i];
    }

    private static int branchedCalculation(int i) {
        if (i == 0) {
            return 1;
        } else if (i == 1) {
            return 2;
        } else if (i == 2) {
            return 3;
        } else {
            return 4;
        }
    }

    private static void timeLogging(String name, Calculator func) {
        long start = System.nanoTime();
        int x = 0;
        for (int i = 0; i < 1000000000; i++) {
            x += func.calculate(1);
        }
        long end = System.nanoTime();
        long duration = (end - start) / 1000000; // Convert to milliseconds
        System.out.printf("%s: %dms (x=%d)%n", name, duration, x);
    }

    @FunctionalInterface
    interface Calculator {
        int calculate(int i);
    }

    public static void main(String[] args) {
        timeLogging("branched_calculation", Main::branchedCalculation);
        timeLogging("branchless_calculation", Main::branchlessCalculation);
    }
}
