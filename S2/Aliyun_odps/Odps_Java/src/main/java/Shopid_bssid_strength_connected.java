import com.aliyun.odps.udf.ExecutionContext;
import com.aliyun.odps.udf.UDFException;
import com.aliyun.odps.udf.UDTF;
import com.aliyun.odps.udf.annotation.Resolve;

// TODO define input and output types, e.g. "string,string->string,bigint".
@Resolve({"string,string->string,string,string,string"})
public class Shopid_bssid_strength_connected extends UDTF {

    @Override
    public void setup(ExecutionContext ctx) throws UDFException {

    }

    @Override
    public void process(Object[] args) throws UDFException {
        // TODO
        String shop_id = (String) args[0];
        String wifi_infos = (String) args[1];
        String bssid = "0";
        String strength = "0";
        String connected = "false";
        if (wifi_infos.indexOf(";") != -1) {
            String[] wifi_split = wifi_infos.split(";");
            //wifi_infos not connected
            for (int i = 0; i < wifi_split.length; i++) {
                bssid = wifi_split[i].split("\\|")[0];
                strength = wifi_split[i].split("\\|")[1];
                connected = wifi_split[i].split("\\|")[2];
                forward(shop_id, bssid, strength, connected);
            }
        } else {
            bssid = wifi_infos.split("\\|")[0];
            strength = null;
            connected = "true";
            forward(shop_id, bssid, strength, connected);
        }

    }

    @Override
    public void close() throws UDFException {

    }

}