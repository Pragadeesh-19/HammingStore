package io.hammingstore.spring.data.annotation;

import java.lang.annotation.*;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Documented
public @interface CircuitBreaker {

    String name() default "hammingstore";

    String fallback() default "throw";
}
