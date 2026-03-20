package io.hammingstore.spring.data.annotation;

import java.lang.annotation.*;

@Target(ElementType.MODULE)
@Retention(RetentionPolicy.RUNTIME)
@Documented
public @interface HopQuery {

    String value();
}
