#pragma once

namespace hyle::test_support {

// Self-signed CN=localhost certificate used only by provider-on loopback
// integration tests. Keeping the fixture in one support header lets protocol
// factors exercise real TLS without invoking an external openssl executable.
inline constexpr char kTlsCertificatePem[] =
    "-----BEGIN CERTIFICATE-----\n"
    "MIIDCTCCAfGgAwIBAgIUN8o8u2IFjiNjqkZO2bW2LU5tXXowDQYJKoZIhvcNAQEL\n"
    "BQAwFDESMBAGA1UEAwwJbG9jYWxob3N0MB4XDTI2MDcwNDA3MTA1M1oXDTM2MDcw\n"
    "MTA3MTA1M1owFDESMBAGA1UEAwwJbG9jYWxob3N0MIIBIjANBgkqhkiG9w0BAQEF\n"
    "AAOCAQ8AMIIBCgKCAQEAqdKC1htqD9qXhLF2wDBn565oY4rs6rP2iCyTDfO5ePP/\n"
    "sLOCzDkiVza1MjhwSbuK6zDRbrY2joX3Ir6KLbzoV/5VaVckBocPTcNxQa42DmIH\n"
    "7xZoDRu9pstp8zdpy3UfAruPRyNWqHqVnw09DQETLbwh3blhT1PaMAOMIBJGCOnd\n"
    "/NsBqiVOsYAkNVgcbuqV8h9b3b6FMMoPPRCVnfn5XROLk5Bh47zT+Q1WAsqyARDe\n"
    "WZ5q8TKQzHuFSursB8rB9kYZLHXmGnnTe2dsiypqqWyCOWryxchEL9mKEk2Q2UUc\n"
    "7VDtvOxgZKu4GJ1r9S4UYm4nB4OPqdIJ96EE+QcIYwIDAQABo1MwUTAdBgNVHQ4E\n"
    "FgQUZIsMaR+5CUbVAhXUhUBDh1+ubNIwHwYDVR0jBBgwFoAUZIsMaR+5CUbVAhXU\n"
    "hUBDh1+ubNIwDwYDVR0TAQH/BAUwAwEB/zANBgkqhkiG9w0BAQsFAAOCAQEAFvOf\n"
    "AcST8Bmpf26HQCnLYCjZfrvw7XIsoJ08KT9JVx8PXXnXsfKB/n8ZlfU7GMviMIYI\n"
    "c27FLOzb8hA2Ifw8wcRW3sC9rEReblI8+bwPchcexB3jyAdf13zcN4GFBrUy3j91\n"
    "1fdlNBllsgizWy7fF5ARfClcDNM/q4t/KcYfSpwcWebXJc6B7IdfDGLcMJ5NzR4C\n"
    "DzAn6Ks4CLxOXJVq2A/h14uUlyS9lPMps2zQ2ggeEg4tOKgln0BRBkiUBKwgBHAo\n"
    "i4W6kCNLXKk2OG5WP/A+F3FukS9r5sbGgdQC5ANxndDwfSUuYcEYDRn+P7mdEbBZ\n"
    "+9hlP09t1qimUNJa/g==\n"
    "-----END CERTIFICATE-----\n";

inline constexpr char kTlsPrivateKeyPem[] =
    "-----BEGIN PRIVATE KEY-----\n"
    "MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCp0oLWG2oP2peE\n"
    "sXbAMGfnrmhjiuzqs/aILJMN87l48/+ws4LMOSJXNrUyOHBJu4rrMNFutjaOhfci\n"
    "vootvOhX/lVpVyQGhw9Nw3FBrjYOYgfvFmgNG72my2nzN2nLdR8Cu49HI1aoepWf\n"
    "DT0NARMtvCHduWFPU9owA4wgEkYI6d382wGqJU6xgCQ1WBxu6pXyH1vdvoUwyg89\n"
    "EJWd+fldE4uTkGHjvNP5DVYCyrIBEN5ZnmrxMpDMe4VK6uwHysH2RhksdeYaedN7\n"
    "Z2yLKmqpbII5avLFyEQv2YoSTZDZRRztUO287GBkq7gYnWv1LhRibicHg4+p0gn3\n"
    "oQT5BwhjAgMBAAECggEAAIkdvFLwaTKW/4ykI4pwV9s4qhCCNPuyhdugiE7VLjjH\n"
    "lpH7+IZSzoEC9+Phesh0JJzBTfH8k90ZTYtAGh46T18jSGzfCLdvu9H0SMqlxTLX\n"
    "wPaXOmwn4EXlxIyCsWkVQXSmlYj0TFC8QxJ1t4iu1YCX5lnTOqUfKL9ohDtg7CtE\n"
    "8dXYQ6tC6vyc8YdFpt9NoEcBxmy3voIZw37z1f1bVQAVIo47vq1K/a0rHhqXtJPc\n"
    "bjbpd1EWbCkR/1Azlw4AIIScJLHHsIvpUlP3xg30qNKkxp5LIg8dfA3vihmEArQ/\n"
    "dJPv1xn6ySkoKFRooPOwQRtqcrPeVUVSI3Pd8sCR/QKBgQDWXfiOe29tRkydBvL4\n"
    "Ktlh5u9PPxwTrmTmGwoh50H5XBkL/3+Oz5cNGVSGdzES/3+oZwFkZgshM/2Fvo83\n"
    "uYui/76G8HhK8B7duEH4kGieyki5kVInq7gBUbpCsZrhbv3b1NndXcnJUUkkM+5f\n"
    "3jvzCHTaaxy3Fgb1RfqNES/DfwKBgQDKzdZQHZZQk8k5+IAqZ0vpI0w4606iha8C\n"
    "vk4fUapZejXnMcRobiP1etVtfVg3YTjFZb3+dSp7zCWe9v7P2tRCm/LfwnbAiRmb\n"
    "LavUTtDD2RE1Mnnz6FbtE5nGZ7BxdxYiuF3Gof1zW0ooCSNFW4m3VymIaHzY5nV9\n"
    "AmTzPMedHQKBgFk80UXvGJmy1B3iUK8Cl2/sBN9GdFe1F4e65eLrNEmsDzkxaVZ4\n"
    "EHeBffBkGefAuNHP34J+AFAYJofQpioacYivUg82d70qbI78L2pcI1FHADNPzZV6\n"
    "JKVX8mPlL4Z8/+zNlMxUSLYnwnzxqV/7UZWAhbSPNmr5HnhoYz7qUqndAoGBAL/q\n"
    "Hb7e4NoLoEcNlCuMaNeMoZrTgixmGYGdGUyl+XCHJ+9KtzSj1b99p+7eyNZPWGvk\n"
    "pfDga32PaKKP1f5YdLk73pBE9B+rHeItTAezNfe/uHsA80sPOQNMox6QDPay3O2q\n"
    "ToOj07aQUxKj4w7FWwN2pkUBq/ON0uyjkWhkwfk5AoGBALKE/2Nc8VAIlhcnqZDJ\n"
    "SRROsPLt1I56SykkMB9H3OhH3jW2xaM043IcqyS8GAXAXBS4ltZ3vYzyy+dkoKHb\n"
    "1/HH0oyQh5LdX6vLMtXxoZI8GPnmtbeJ4fmXckbRXev925vXgdcknQ2RY6OU9HzI\n"
    "A48heHgTiFQ6QIIrkn9z7lQJ\n"
    "-----END PRIVATE KEY-----\n";

} // namespace hyle::test_support
