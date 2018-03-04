import com.aliyun.odps.udf.ExecutionContext;
import com.aliyun.odps.udf.UDFException;
import com.aliyun.odps.udf.UDTF;
import com.aliyun.odps.udf.annotation.Resolve;

import java.text.DecimalFormat;
import java.util.HashMap;

import static java.util.stream.Collectors.joining;
import static jdk.nashorn.internal.objects.NativeMath.round;

// TODO define input and output types, e.g. "string,string->string,bigint".
@Resolve({"bigint,Double,Double,string->string,string"})
public class train_mall_kv extends UDTF {
    public static HashMap<Integer, Integer> bssid_index = new HashMap<>();

    @Override
    public void setup(ExecutionContext ctx) throws UDFException {

    }

    @Override
    public void process(Object[] args) throws UDFException {

        // TODO use
        String shop_ix_in_mall = String.valueOf((Long) args[0]);
        String longitude = String.valueOf((Double) args[1]);
        String latitude = String.valueOf((Double) args[2]);
        String wifi_infos = (String) args[3];
        HashMap<Integer, Double> hashMap = new HashMap<Integer, Double>();
        DecimalFormat df = new DecimalFormat("#.00");
        String bssid = "0";
        String strength = "0";
        hashMap.put(0, Double.parseDouble(df.format(Double.parseDouble(longitude))));
        hashMap.put(1, Double.parseDouble(df.format(Double.parseDouble(latitude))));
        if (wifi_infos.indexOf(";") != -1) {
            String[] wifi_split = wifi_infos.split(";");
            //wifi_infos not connected
            for (int i = 0; i < wifi_split.length; i++) {
                bssid = wifi_split[i].split("\\|")[0];
                strength = wifi_split[i].split("\\|")[1];
                hashMap.put(Integer.parseInt(bssid.split("_")[1]), Double.parseDouble(df.format(Double.parseDouble(strength))));
                String s = hashMap.entrySet()
                        .stream()
                        .map(e -> e.getKey() + ":" + e.getValue())
                        .collect(joining(" "));
                forward(shop_ix_in_mall, "\"" + s + "\"");

            }
        } else {
            bssid = wifi_infos.split("\\|")[0];
            hashMap.put(Integer.parseInt(bssid.split("_")[1]), Double.parseDouble(df.format(-35)));
            String s = hashMap.entrySet()
                    .stream()
                    .map(e -> e.getKey() + ":" + e.getValue())
                    .collect(joining(" "));
            forward(shop_ix_in_mall, "\"" + s + "\"");

        }
    }

    @Override
    public void close() throws UDFException {

    }

}