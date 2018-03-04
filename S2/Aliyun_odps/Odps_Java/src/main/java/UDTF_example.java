import com.aliyun.odps.udf.ExecutionContext;
import com.aliyun.odps.udf.UDFException;
import com.aliyun.odps.udf.UDTF;
import com.aliyun.odps.udf.annotation.Resolve;

import java.util.HashMap;

// TODO define input and output types, e.g. "string,string->string,bigint".
@Resolve({"string,string->string,string,string"})
public class UDTF_example extends UDTF {

    @Override
    public void setup(ExecutionContext ctx) throws UDFException {

    }

    @Override
    public void process(Object[] args) throws UDFException {
        // TODO
        //Object[] contains shop_id and wifi_infos
        String shop_id = (String) args[0];
        String wifi_infos = (String) args[1];
        String bssid = "0";
        String strength = "0";
        if (wifi_infos.indexOf(";") != -1) {
            String[] wifi_split = wifi_infos.split(";");
            //wifi_infos not connected
            for (int i = 0; i < wifi_split.length; i++) {
                bssid = wifi_split[i].split("\\|")[0];
                strength = wifi_split[i].split("\\|")[1];
                forward(shop_id, bssid, strength);
            }
        } else {
            bssid = wifi_infos.split("\\|")[0];
            strength = "-1";
            forward(shop_id, bssid, strength);
        }
    }


    @Override
    public void close() throws UDFException {

    }

}