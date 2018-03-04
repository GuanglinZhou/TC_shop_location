import com.aliyun.odps.udf.UDF;
import com.aliyun.odps.udf.UDFException;
import com.aliyun.odps.udf.UDTF;

import java.util.*;

//输入train_test总表，输出每个mall的wifi向量表

public class wifi_example extends UDF {
    public static int train_len = 10913285;
    public static int test_len = 2402119;
    public static HashMap<String, Integer> bssid_num = new HashMap<String, Integer>();

    List<Object[]> x = new ArrayList<Object[]>();

    public String evaluate(String s) {
//        bssid_num.put(new Random().nextInt() + "", 123);
//        Object[] y = new Object[1024 * 1024 * 51];
//        x.add(y);
        return "hello world:" + Runtime.getRuntime().freeMemory();
    }
//    public String evaluate(String s) {
//
//        return "hello world:" + bssid_num.size();
//    }

//    //    todo split the train and test data
//    public void process(Object[] objects) throws UDFException {
//        System.out.println("Start constructing wifi vector...");
//        String wifi_infos = (String) objects[5];
//        if (wifi_infos.indexOf(";") != -1) {
//            String[] wifi_split = wifi_infos.split(";");
//            //wifi_infos not connected
//            for (int i = 0; i < wifi_split.length; i++) {
//                bssid_num.put(wifi_split[i].split("\\|")[0], bssid_num.get(wifi_split[i].split("\\|")[0]) + 1);
//            }
//        } else {
//            //wifi_infos is connected
//            bssid_num.put(wifi_infos.split("\\|")[0], 10);
//        }
//        System.out.println(evaluate("123"));
//
//    }


}