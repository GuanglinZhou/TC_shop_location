import com.aliyun.odps.udf.UDF;
import com.aliyun.odps.udf.UDFException;
import com.aliyun.odps.udf.UDTF;
import com.aliyun.odps.udf.annotation.Resolve;

import java.util.*;

//输入train_test总表，输出每个mall的wifi向量表
@Resolve({"string,string->string,bigint"})
public class Wifi_Vector extends UDTF {
    public static int train_len = 10913285;
    public static int test_len = 2402119;
    public Wifi_Vector() {
        System.out.println("Wifi_Vector constructor...");
    }
    //    todo split the train and test data
    @Override
    public void process(Object[] objects) throws UDFException {
        System.out.println("Start constructing wifi vector...");
        HashMap<String, HashMap<String, Integer>> mallid_bssid_num = new HashMap<String, HashMap<String, Integer>>();
        String index = (String) objects[0];
        String wifi_infos = (String) objects[5];
        String mall_id = (String) objects[2];
        if (wifi_infos.indexOf(";") != -1) {
            String[] wifi_split = wifi_infos.split(";");
            //wifi_infos not connected
            for (int i = 0; i < wifi_split.length; i++) {
                HashMap<String, Integer> bssid_num = new HashMap<String, Integer>();
                bssid_num.put(wifi_split[i].split("\\|")[0], bssid_num.get(wifi_split[i].split("\\|")[0]) + 1);
                mallid_bssid_num.put(mall_id, bssid_num);
            }
        } else {
            //wifi_infos is connected
            HashMap<String, Integer> bssid_num = new HashMap<String, Integer>();
            bssid_num.put(wifi_infos.split("\\|")[0], 10);
            mallid_bssid_num.put(mall_id, bssid_num);
        }
        System.out.println("There are " + mallid_bssid_num.keySet().size()+" mall_id");
//        forward(a, b);
    }

//    public static void printMap(Map mp) {
//        Iterator it = mp.entrySet().iterator();
//        while (it.hasNext()) {
//            Map.Entry pair = (Map.Entry) it.next();
//            System.out.println(pair.getKey() + " = " + pair.getValue());
//            it.remove(); // avoids a ConcurrentModificationException
//        }
//    }
//    public HashMap<String, Integer> bsssid_num_map(String s) {
//        HashMap<String, Integer> hashMap = new HashMap<String, Integer>();
//
//        return hashMap;
//
//    }
}