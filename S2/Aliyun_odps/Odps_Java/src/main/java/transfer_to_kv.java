import com.aliyun.odps.udf.ExecutionContext;
import com.aliyun.odps.udf.UDFException;
import com.aliyun.odps.udf.UDTF;
import com.aliyun.odps.udf.annotation.Resolve;

// TODO define input and output types, e.g. "string,string->string,bigint".
@Resolve({"String String String String String String String String String String String String String String String String String String String String String String String String String String String String String String String String String -> String "})
public class transfer_to_kv extends UDTF {

    @Override
    public void setup(ExecutionContext ctx) throws UDFException {

    }

    @Override
    public void process(Object[] args) throws UDFException {
        // mall_id="m_7701"
        

    }

    @Override
    public void close() throws UDFException {

    }

}